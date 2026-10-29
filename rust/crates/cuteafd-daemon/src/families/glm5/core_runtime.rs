//! GLM's native K6 projection seams over shared compressed-weight owners.
use crate::shared::{dense_trellis::{CoreWeights, Projection, ProjectionLayout, ProjectionScratch},
    memory::device::{Allocation, Device}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, dense_trellis::TrellisDType, programs::Programs,
    F32Add, V41PeerCopy};
use std::{collections::BTreeMap, ffi::c_void};

/// Every actual header-selected geometry is resolved before capture. Projection
/// scratch is serialized; the caller retains split output cores in row slots.
pub(crate) struct CorePrograms<'a> {
    projections: BTreeMap<(usize, usize, usize, &'static str), Projection<'a>>,
    layouts: [ProjectionLayout; 2],
    stages: [cuteafd_loader::families::glm5::core::CoreStageLayout; 2],
    pub add: F32Add<'a>,
    pub copy: V41PeerCopy<'a>,
}
impl<'a> CorePrograms<'a> {
    pub(crate) fn new(programs: &'a Programs<'a>, cfg: &cuteafd_loader::families::glm5::GlmDsaConfig,
        layers: &[super::weights::GlmLayer<'_>], unit: usize, ranks: usize,
        admitted: Option<&cuteafd_loader::families::glm5::core::CoreRankLayout>) -> Result<Option<Self>> {
        let rows = unit.max(super::engine::DECODE_ROWS);
        let mut projections = BTreeMap::new();
        let mut layouts: [Option<ProjectionLayout>; 2] = [None, None];
        for weights in layers.iter().flat_map(|layer| layer.core.values()) {
            let key = projection_key(weights);
            if projections.contains_key(&key) { continue; }
            let projection = Projection::load_geometry(programs, key.0, key.1, key.2, key.3,
                rows, TrellisDType::Bf16)?;
            ProjectionLayout::new(&projection.program, rows)?;
            for (layout, rows) in layouts.iter_mut().zip([super::engine::DECODE_ROWS, unit]) {
                let required = cuteafd_loader::formats::trellis_core::ProjectionScratchLayout::new(key.0, key.1, rows)?;
                let required = ProjectionLayout { rows, extents: required.extents };
                if let Some(layout) = layout { layout.merge(&required)?; } else { *layout = Some(required); }
            }
            projections.insert(key, projection);
        }
        if projections.is_empty() {
            ensure!(admitted.is_none_or(|a| a.body.is_empty()), "GLM K6 admitted body has no loaded owners");
            return Ok(None);
        }
        let admitted = admitted.context("GLM K6 body has no admitted native layout")?;
        let layouts = [layouts[0].take().context("GLM K6 decode layout")?,
            layouts[1].take().context("GLM K6 prefill layout")?];
        for (actual, expected) in layouts.iter().zip([admitted.decode.as_ref(), admitted.prefill.as_ref()]) {
            let expected = expected.context("GLM K6 admitted projection layout")?;
            ensure!(actual.rows == expected.rows && actual.extents == expected.extents,
                "GLM K6 loaded body scratch differs from admission");
        }
        // Neutral projection and GLM stage namespaces are distinct; both must
        // resolve on this rank before an unmatched peer wait or graph capture.
        programs.load_matching(|name| stage_program(name, ranks))?;
        let owned = layers.iter().any(|layer| layer.attention_owner.is_some());
        for inter in [cfg.dense_intermediate / ranks, cfg.moe_intermediate * cfg.shared_experts / ranks] {
            let prefix = if ranks == 2 { "glm2" } else { "glm" };
            programs.program(&format!("{prefix}_stage_swiglu_i{inter}"), &["gate", "up", "hidden"])?;
        }
        let stages = [cuteafd_loader::families::glm5::core::CoreStageLayout::new(cfg, super::engine::DECODE_ROWS, ranks, owned)?,
            cuteafd_loader::families::glm5::core::CoreStageLayout::new(cfg, unit, ranks, owned)?];
        ensure!(admitted.stages.as_ref() == Some(&stages), "GLM K6 loaded BF16 stages differ from admission");
        Ok(Some(Self { projections, layouts, stages,
            add: programs.library().f32_add()?, copy: programs.library().v41_peer_copy()? }))
    }
    pub(crate) fn projection(&self, weights: &CoreWeights<'_>) -> Result<&Projection<'a>> {
        self.projections.get(&projection_key(weights)).context("GLM K6 geometry not bound before capture")
    }
    pub(crate) fn allocate(&self, device: Device<'a>, rows: usize, decode: bool) -> Result<CoreScratch<'a>> {
        let index = usize::from(!decode);
        let layout = &self.layouts[index];
        ensure!(rows == layout.rows, "GLM K6 scratch differs from admitted row capacity");
        let projection = layout.allocate(device)?;
        let stages = self.stages[index].extents.iter().map(|(&name, &bytes)|
            Ok((name, Allocation::new(device, bytes)?))).collect::<Result<_>>()?;
        Ok(CoreScratch { projection, stages })
    }
}

pub(crate) struct CoreScratch<'a> {
    pub projection: ProjectionScratch<'a>,
    stages: BTreeMap<&'static str, Allocation<'a>>,
}
impl CoreScratch<'_> {
    pub(crate) fn stage(&self, name: &str) -> Result<Buffer> {
        Ok(self.stages.get(name).with_context(|| format!("GLM K6 missing stage {name}"))?.buffer)
    }
}
fn stage_program(name: &str, ranks: usize) -> bool {
    cuteafd_loader::families::glm5::DensePrograms::Trellis.stage(name, ranks == 2)
}
fn projection_key(weights: &CoreWeights<'_>) -> (usize, usize, usize, &'static str) {
    use cuteafd_loader::formats::dense_trellis::TrellisSide;
    (weights.shape[1], weights.shape[0], if weights.side == TrellisSide::Input { 2 } else { 1 }, book(weights))
}

pub(crate) use crate::shared::dense_trellis::{CoreHead, head_launch};

/// Copy only logical columns after the complete stored-N transform has run.
pub(crate) fn pack_layout(rows: usize, stored: usize, logical: usize, pitch: usize, column: usize)
    -> Result<(usize, usize, usize, usize)> {
    ensure!(rows > 0 && stored >= logical && logical > 0
        && column.checked_add(logical).is_some_and(|end| end <= pitch),
        "GLM K6 logical crop/pack geometry");
    let bytes = |n: usize| n.checked_mul(2).ok_or_else(|| anyhow::anyhow!("GLM K6 pack byte overflow"));
    let source = rows.checked_mul(stored).ok_or_else(|| anyhow::anyhow!("GLM K6 source extent overflow"))?;
    let target = rows.checked_mul(pitch).ok_or_else(|| anyhow::anyhow!("GLM K6 target extent overflow"))?;
    bytes(source)?; bytes(target)?;
    Ok((bytes(stored)?, bytes(pitch)?, bytes(logical)?, bytes(column)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_target_head_rejects_concurrent_foreign_stream_before_scratch_use() -> Result<()> {
        let bound = std::ptr::without_provenance_mut::<c_void>(1);
        let peer = std::ptr::without_provenance_mut::<c_void>(2);
        let unit = std::ptr::without_provenance_mut::<c_void>(3);
        // Pending GPU users do not finish between these host hooks. On the
        // bound stream their execution is FIFO; a foreign stream could race.
        let mut pending = Vec::new();
        for (path, rows, stream) in [("decode", 1, bound), ("prefill", 64, bound),
            ("prefill-peer", 32, peer), ("decode-unit", 1, unit), ("decode", 20, bound)] {
            let result = head_launch(64, bound, rows, stream, || {
                pending.push((path, stream));
                Ok(())
            });
            assert_eq!(result.is_ok(), stream == bound);
        }
        assert_eq!(pending.iter().map(|(path, _)| *path).collect::<Vec<_>>(), ["decode", "prefill", "decode"]);
        assert!(pending.iter().all(|(_, stream)| *stream == bound));
        for rows in [0, 65] {
            assert!(head_launch(64, bound, rows, bound, || -> Result<()> {
                panic!("invalid rows reached scratch")
            }).is_err());
        }
        Ok(())
    }

    #[test]
    fn eager_stages_stay_in_the_admitted_rank_namespaces() {
        let manifest = serde_json::json!({"programs": [
            {"name": "glm_stage_rank_pack_m64"},
            {"name": "glm_stage_swiglu_i2048"},
            {"name": "glm2_stage_rank_pack_m64"},
            {"name": "glm2_stage_swiglu_i1024"},
            {"name": "dense_trellis_k6_input_k6144"},
            {"name": "kimi_stage_rank_pack_m64"}
        ]});
        for ranks in [1, 2] {
            let admitted = cuteafd_loader::placement::ProgramSet::from_manifest(&manifest,
                if ranks == 2 { &["glm", "glm2", "glm_stage", "glm2_stage"] } else { &["glm", "glm_stage"] });
            let loaded = manifest["programs"].as_array().unwrap().iter()
                .filter_map(|p| p["name"].as_str()).filter(|name| stage_program(name, ranks)).collect::<Vec<_>>();
            assert_eq!(loaded.len(), ranks * 2);
            assert!(loaded.iter().all(|name| admitted.names.iter().any(|p| p == name)));
        }
        assert!(!stage_program("glm2_stage_swiglu_i1024", 1));
        assert!(stage_program("glm2_stage_swiglu_i1024", 2));
    }

    #[test]
    fn split_shared_core_rounds_once_and_contributes_once() {
        let narrow = |x: f32| {
            let bits = x.to_bits();
            f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
        };
        let h128 = |mut x: Vec<f32>| {
            for width in [1, 2, 4, 8, 16, 32, 64] {
                for block in (0..128).step_by(width * 2) {
                    for offset in 0..width {
                        let (a, b) = (x[block + offset], x[block + offset + width]);
                        x[block + offset] = a + b;
                        x[block + offset + width] = a - b;
                    }
                }
            }
            x
        };
        let mut extra_rounding_changes = 0;
        for row in 0..8 {
            let first = (0..128).map(|n| (((n * 37 + row * 13) % 257) as f32 - 128.0) / 29.0).collect::<Vec<_>>();
            let second = (0..128).map(|n| (((n * 61 + row * 17) % 251) as f32 - 125.0) / 31.0).collect::<Vec<_>>();
            let scales = (0..128).map(|n| (n % 7 + 1) as f32 / 16.0).collect::<Vec<_>>();
            // A whole projection accumulates the two contiguous K halves in
            // rank order, then performs one output transform and BF16 egress.
            let reference = h128(first.iter().zip(&second).map(|(a, b)| a + b).collect());
            let shared = reference.iter().zip(&scales).map(|(x, s)| narrow(x * s)).collect::<Vec<_>>();
            let reduced = first.iter().zip(&second).map(|(a, b)| a + b).collect::<Vec<_>>();
            let actual = h128(reduced).iter().zip(&scales).map(|(x, s)| narrow(x * s)).collect::<Vec<_>>();
            let a = h128(first);
            let b = h128(second);
            for n in 0..128 {
                let routed = narrow(((n * 19 + row) % 101) as f32 / 11.0);
                let single = narrow(routed + shared[n]);
                let rank0 = narrow(routed + actual[n]);
                let rank1_shared = 0.0f32;
                assert_eq!(narrow(rank0 + rank1_shared).to_bits(), single.to_bits());
                let lossy = narrow(narrow(a[n] * scales[n]) + narrow(b[n] * scales[n]));
                extra_rounding_changes += usize::from(lossy.to_bits() != shared[n].to_bits());
            }
        }
        assert!(extra_rounding_changes > 0, "fixture must detect per-rank BF16 egress");
        // This algebraic test is not a substitute for the native GPU golden.
        let source = include_str!("engine/native_dense.rs");
        let meet = source.split("fn meet_shared_core(").nth(1).unwrap()
            .split("/// Input-side projections").next().unwrap();
        assert!(meet.contains("if !Self::core_split(layer, \"down\") { return Ok(()); }"));
        assert!(meet.contains("if rank == 0") && meet.contains("cuda_zero_bytes_async"));
        let reduce = source.split("fn reduce_projection(").nth(1).unwrap()
            .split("/// The caller").next().unwrap();
        assert!(reduce.contains("if rank == 0 { (local, peer) } else { (peer, local) }"));
        assert!(reduce.contains(".reduced(&programs.add"));
    }

    #[test]
    fn full_padded_transform_is_cropped_only_when_packing_qkv_rows() -> Result<()> {
        for rows in [1, 8, 64, 4096] {
            let (src_pitch, dst_pitch, width, offset) = pack_layout(rows, 640, 576, 2624, 2048)?;
            assert_eq!((src_pitch, dst_pitch, width, offset), (1280, 5248, 1152, 4096));
            let full = (0..rows * 640).map(|i| (i ^ (i >> 8)) as u16).collect::<Vec<_>>();
            let mut packed = vec![0xDEADu16; rows * 2624];
            for r in 0..rows {
                packed[r * 2624 + 2048..][..576].copy_from_slice(&full[r * 640..][..576]);
            }
            for r in 0..rows {
                assert_eq!(&packed[r * 2624 + 2048..][..576], &full[r * 640..][..576]);
                assert!(packed[r * 2624..][..2048].iter().all(|&v| v == 0xDEAD));
            }
        }
        assert_eq!(pack_layout(1, 2048, 2048, 2624, 0)?, (4096, 5248, 4096, 0));
        for args in [(0, 640, 576, 2624, 2048), (1, 576, 640, 2624, 0),
            (1, 640, 576, 2624, 2049), (usize::MAX, 640, 576, 2624, 2048)] {
            assert!(pack_layout(args.0, args.1, args.2, args.3, args.4).is_err());
        }
        Ok(())
    }
}

pub(crate) fn book(weights: &CoreWeights<'_>) -> &'static str {
    match weights.codebook {
        cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mcg => "mcg",
        cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mul1 => "mul1",
    }
}
