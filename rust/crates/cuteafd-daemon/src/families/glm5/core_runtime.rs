//! GLM's native K6 projection seams over shared compressed-weight owners.
use crate::shared::{dense_trellis::{CoreWeights, Projection, ProjectionLayout, ProjectionScratch},
    memory::{device::{Allocation, Device}, Tag}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, dense_trellis::TrellisDType, programs::Programs,
    Bf16F32Add, F32Bf16Narrow, V41PeerCopy};
use std::{collections::BTreeMap, ffi::c_void};

/// Every actual header-selected geometry is resolved before capture. Projection
/// scratch is serialized; the caller retains split output cores in row slots.
pub(crate) struct CorePrograms<'a> {
    projections: BTreeMap<(usize, usize, usize, &'static str, usize), Projection<'a>>,
    layouts: [Option<ProjectionLayout>; 2],
    stages: [cuteafd_loader::families::glm5::core::CoreStageLayout; 2],
    native_bf16: [cuteafd_ffi::draft_projection::DraftProjections<'a>; 2],
    native_fp8: [[cuteafd_ffi::native_attention::Fp8Projections<'a>; 2]; 2],
    native_scratch: [usize; 2],
    native_programs: Vec<String>,
    pub add: Bf16F32Add<'a>,
    pub narrow: F32Bf16Narrow<'a>,
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
            let projection = Projection::load_geometry_bits(programs, key.0, key.1, key.2, key.4, key.3,
                rows, TrellisDType::Bf16)?;
            ProjectionLayout::new(&projection.program, rows)?;
            let fused = projection.program.has_fused();
            for (layout, rows) in layouts.iter_mut().zip([super::engine::DECODE_ROWS, unit]) {
                let required = cuteafd_loader::formats::trellis_core::ProjectionScratchLayout::new_with_fused(
                    key.0, key.1, rows, fused)?;
                let required = ProjectionLayout { rows, extents: required.extents };
                if let Some(layout) = layout { layout.merge(&required)?; } else { *layout = Some(required); }
            }
            projections.insert(key, projection);
        }
        if projections.is_empty() && layers.iter().all(|layer| layer.attention_native.is_empty()) {
            ensure!(admitted.is_none_or(|a| a.body.is_empty()), "GLM K6 admitted body has no loaded owners");
            return Ok(None);
        }
        let admitted = admitted.context("GLM K6 body has no admitted native layout")?;
        let layouts = [layouts[0].take(), layouts[1].take()];
        for (actual, expected) in layouts.iter().zip([admitted.decode.as_ref(), admitted.prefill.as_ref()]) {
            ensure!(actual.as_ref().map(|layout| (layout.rows, &layout.extents))
                == expected.map(|layout| (layout.rows, &layout.extents)),
                "GLM K6 loaded body scratch differs from admission");
        }
        use cuteafd_ffi::draft_projection::{DraftProjections, ProjectionOutput};
        use cuteafd_ffi::native_attention::Fp8Projections;
        use cuteafd_loader::{families::glm5::core::native_output,
            placement::planner::native_attention::{NativeProgram, NativePrograms}};
        let native_layouts = layers.iter().flat_map(|layer| layer.attention_native.values())
            .map(|weight| &weight.layout).collect::<Vec<_>>();
        let requests = native_layouts.iter().map(|&layout| NativeProgram { layout,
            output: native_output(layout) }).collect::<Vec<_>>();
        let mut native_programs = std::collections::BTreeSet::new();
        for capacity in [super::engine::DECODE_ROWS, unit] {
            let selected = NativePrograms::resolve(&requests, capacity, None)?;
            for name in &selected.programs {
                ensure!(programs.contains(name), "GLM native projection export {name} missing before capture");
            }
            native_programs.extend(selected.programs);
        }
        let native_programs = native_programs.into_iter().collect();
        let shapes = |fp8, f32| native_layouts.iter()
            .filter(|layout| layout.fp8 == fp8 && (native_output(layout) == ProjectionOutput::F32) == f32)
            .map(|layout| (layout.shape[1], layout.shape[0])).collect::<Vec<_>>();
        let native_bf16 = [DraftProjections::load_with_output(programs, &shapes(false, false), rows, ProjectionOutput::Bf16)?,
            DraftProjections::load_with_output(programs, &shapes(false, true), rows, ProjectionOutput::F32)?];
        let fp8 = |capacity| -> Result<_> { Ok([
            Fp8Projections::load(programs, &shapes(true, false), capacity, ProjectionOutput::Bf16)?,
            Fp8Projections::load(programs, &shapes(true, true), capacity, ProjectionOutput::F32)?]) };
        let native_fp8 = [fp8(super::engine::DECODE_ROWS)?, fp8(unit)?];
        let mut native_scratch = [0usize; 2];
        for (index, group) in native_fp8.iter().enumerate() {
            for (output, registry) in group.iter().enumerate() {
                for (k, n) in shapes(true, output == 1) {
                    native_scratch[index] = native_scratch[index].max(registry.scratch_bytes(k, n)?);
                }
            }
        }
        ensure!(native_scratch == admitted.native_scratch, "GLM native scratch differs from admission");
        let mut actual = layers.iter().flat_map(|layer| layer.attention_native.values())
            .map(|weight| weight.layout.clone()).collect::<Vec<_>>();
        let mut expected = admitted.native.clone();
        actual.sort_by(|a,b| a.stem.cmp(&b.stem)); expected.sort_by(|a,b| a.stem.cmp(&b.stem));
        ensure!(actual == expected, "GLM native projection owners differ from admission");
        // Neutral projection and GLM stage namespaces are distinct; both must
        // resolve on this rank before an unmatched peer wait or graph capture.
        programs.load_matching(|name| stage_program(name, ranks))?;
        let owned = layers.iter().any(|layer| layer.attention_owner.is_some());
        for inter in [cfg.dense_intermediate / ranks, cfg.moe_intermediate * cfg.shared_experts / ranks] {
            let name = if cfg.variant.is_glm() {
                format!("{}_stage_swiglu_i{inter}", if ranks == 2 { "glm2" } else { "glm" })
            } else { format!("{}_swiglu_i{inter}", cfg.program_family()) };
            programs.program(&name, &["gate", "up", "hidden"])?;
        }
        let stages = [cuteafd_loader::families::glm5::core::CoreStageLayout::new(cfg, super::engine::DECODE_ROWS, ranks, owned)?,
            cuteafd_loader::families::glm5::core::CoreStageLayout::new(cfg, unit, ranks, owned)?];
        ensure!(admitted.stages.as_ref() == Some(&stages), "GLM K6 loaded BF16 stages differ from admission");
        Ok(Some(Self { projections, layouts, stages, native_bf16, native_fp8, native_scratch, native_programs,
            add: programs.library().bf16_f32_add()?, narrow: programs.library().f32_bf16_narrow()?, copy: programs.library().v41_peer_copy()? }))
    }
    /// Include native input/core/output policy entries, including split-K's
    /// row transition. Missing exporter metadata disables exact packing.
    pub(crate) fn packed_programs(&self) -> impl Iterator<Item = &str> {
        self.projections.values().flat_map(|projection| {
            let p = projection.program.policy();
            [p.native_input.as_str(), p.native.as_str(), p.output_program.as_str()]
        }).chain(self.native_programs.iter().map(String::as_str))
    }

    pub(crate) fn projection(&self, weights: &CoreWeights<'_>) -> Result<&Projection<'a>> {
        self.projections.get(&projection_key(weights)).context("GLM K6 geometry not bound before capture")
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn native(&self, weights: &crate::shared::native_attention::NativeAttentionWeights<'_>,
        scratch: &CoreScratch<'_>, input: Buffer, output: Buffer, rows: usize, stream: *mut c_void) -> Result<()> {
        let [n, k] = weights.layout.shape;
        let f32 = usize::from(cuteafd_loader::families::glm5::core::native_output(&weights.layout)
            == cuteafd_ffi::draft_projection::ProjectionOutput::F32);
        // SAFETY: rank-local admitted owners and scratch outlive the ordered stream;
        // typed projection wrappers validate output extents before launch.
        unsafe {
            if weights.layout.fp8 {
                self.native_fp8[usize::from(scratch.prefill)][f32].launch(input.ptr,
                    weights.values.buffer.ptr, weights.scales.as_ref().context("GLM native scales")?.buffer.ptr,
                    output, scratch.native.as_ref().context("GLM native FP8 scratch")?.buffer, rows, k, n, stream)?;
            } else {
                self.native_bf16[f32].launch_checked(input.ptr, weights.values.buffer.ptr, output, rows, k, n, stream)?;
            }
        }
        Ok(())
    }
    pub(crate) fn allocate(&self, device: Device<'a>, rows: usize, decode: bool) -> Result<CoreScratch<'a>> {
        let index = usize::from(!decode);
        let layout = &self.layouts[index];
        let projection = layout.as_ref().map(|layout| {
            ensure!(rows == layout.rows, "GLM K6 scratch differs from admitted row capacity");
            layout.allocate(device, crate::shared::memory::Tag::workspace("glm/native-projection"))
        }).transpose()?;
        let stages = self.stages[index].extents.iter().map(|(&name, &bytes)|
            Ok((name, Allocation::new(device, bytes, Tag::workspace("glm/core-scratch"))?))).collect::<Result<_>>()?;
        let native = (self.native_scratch[index] > 0)
            .then(|| Allocation::new(device, self.native_scratch[index], Tag::workspace("glm/core-scratch"))).transpose()?;
        Ok(CoreScratch { projection, stages, native, prefill: !decode })
    }
}

pub(crate) struct CoreScratch<'a> {
    pub projection: Option<ProjectionScratch<'a>>,
    stages: BTreeMap<&'static str, Allocation<'a>>,
    pub native: Option<Allocation<'a>>,
    prefill: bool,
}
impl CoreScratch<'_> {
    pub(crate) fn allocation_stages(&self) -> impl Iterator<Item = (&'static str, Buffer)> + '_ {
        self.stages.iter().map(|(&name, owner)| (name, owner.buffer))
    }

    pub(crate) fn stage(&self, name: &str) -> Result<Buffer> {
        Ok(self.stages.get(name).with_context(|| format!("GLM K6 missing stage {name}"))?.buffer)
    }
}
fn stage_program(name: &str, ranks: usize) -> bool {
    cuteafd_loader::families::glm5::DensePrograms::Trellis.stage(name, ranks == 2)
}
fn projection_key(weights: &CoreWeights<'_>) -> (usize, usize, usize, &'static str, usize) {
    use cuteafd_loader::formats::dense_trellis::TrellisSide;
    (weights.shape[1], weights.shape[0], if weights.side == TrellisSide::Input { 2 } else { 1 }, book(weights), weights.bits)
}

pub(crate) use crate::shared::dense_trellis::head_launch;

/// GLM retains its own target/drafter scratch, while the FP8 launch is shared
/// with Muse and Step. Both formats enforce same-stream FIFO before reuse.
pub(crate) enum CoreHead<'a> {
    Trellis(crate::shared::dense_trellis::CoreHead<'a>),
    Fp8 { executor: crate::shared::target_head::BlockFp8Head<'a>, scratch: Allocation<'a>,
        device:Device<'a>,stream:*mut c_void },
}
impl<'a> CoreHead<'a> {
    pub(crate) fn new(programs:&'a Programs<'a>,weights:&super::weights::GlmHead<'a>,rows:usize,
        device:Device<'a>,stream:*mut c_void,admitted:&cuteafd_loader::families::glm5::core::CoreRankLayout,
        draft:bool,tag:Tag) -> Result<Self> {
        match weights {
            super::weights::GlmHead::K6(weights) => {
                let layout = if draft { &admitted.draft_head } else { &admitted.target_head }
                    .as_ref().context("GLM trellis head scratch not admitted")?;
                Ok(Self::Trellis(crate::shared::dense_trellis::CoreHead::new(programs,weights,rows,
                    device,stream,layout,tag)?))
            }
            super::weights::GlmHead::BlockFp8(weights) => {
                let executor = device.run(|| crate::shared::target_head::BlockFp8Head::load(programs,weights.layout.shape))?;
                let bytes = if draft { admitted.fp8_draft_head_scratch } else { admitted.fp8_head_scratch }
                    .context("GLM FP8 head scratch not admitted")?;
                ensure!(bytes == executor.scratch_bytes()?,"GLM FP8 head scratch differs from admission");
                Ok(Self::Fp8 { executor,scratch:Allocation::new(device,bytes,tag)?,device,stream })
            }
            _ => anyhow::bail!("GLM native head binding received BF16/row-FP8 head"),
        }
    }
    pub(crate) fn launch(&self,weights:&super::weights::GlmHead<'_>,input:Buffer,output:Buffer,
        rows:usize,stream:*mut c_void) -> Result<()> {
        match (self,weights) {
            (Self::Trellis(head),super::weights::GlmHead::K6(weights)) => head.launch(weights,input,output,rows,stream),
            (Self::Fp8 { executor,scratch,device,stream:bound },super::weights::GlmHead::BlockFp8(weights)) => {
                ensure!(stream == *bound,"GLM shared FP8 head scratch borrowed on foreign stream");
                device.run(|| executor.launch(rows,input,weights,output,scratch.buffer,stream))
            }
            _ => anyhow::bail!("GLM native head binding/storage differs"),
        }
    }
    pub(crate) fn target<'s>(&'s self,weights:&'s super::weights::GlmHead<'_>)
        -> crate::shared::speculation::dflash::TargetHead<'s> {
        crate::shared::speculation::dflash::TargetHead::Launch(Box::new(move |input,logits,rows,stream| {
            let (device,k,n) = match (self,weights) {
                (Self::Trellis(_),super::weights::GlmHead::K6(weights)) =>
                    (weights.payload.buffer.device_id,weights.shape[1],weights.shape[0]),
                (Self::Fp8 { device,.. },super::weights::GlmHead::BlockFp8(weights)) =>
                    (device.id,weights.layout.shape[1],weights.layout.shape[0]),
                _ => anyhow::bail!("GLM borrowed native head storage differs"),
            };
            let buffer = |ptr,bytes| Buffer { ptr,bytes,device_id:device,flags:0 };
            self.launch(weights,buffer(input.cast_mut(),rows*k*2),buffer(logits.cast(),rows*n*4),rows,stream)
        }))
    }
    pub(crate) fn allocation_scratch(&self) -> Option<&ProjectionScratch<'a>> {
        match self { Self::Trellis(head) => Some(head.allocation_scratch()),Self::Fp8 {..} => None }
    }
    pub(crate) fn fp8_scratch(&self) -> Option<Buffer> {
        match self { Self::Fp8 { scratch,..} => Some(scratch.buffer),_=>None }
    }
}


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
    fn moe_shared_down_skips_pre_egress_core_exchange_and_improves_fp32_parity() {
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
        let mut old_sq_err = 0.0f64;
        let mut new_sq_err = 0.0f64;
        for row in 0..8 {
            let c0 = (0..128).map(|n| (((n * 37 + row * 13) % 257) as f32 - 128.0) / 29.0).collect::<Vec<_>>();
            let c1 = (0..128).map(|n| (((n * 61 + row * 17) % 251) as f32 - 125.0) / 31.0).collect::<Vec<_>>();
            let scales = (0..128).map(|n| (n % 7 + 1) as f32 / 16.0).collect::<Vec<_>>();
            let r0 = (0..128).map(|n| (((n * 19 + row * 7) % 101) as f32 - 50.0) / 11.0).collect::<Vec<_>>();
            let r1 = (0..128).map(|n| (((n * 23 + row * 11) % 97) as f32 - 48.0) / 11.0).collect::<Vec<_>>();
            // Exact FP32 reference: r0 + r1 + H128(c0 + c1) * svh.
            let s_fp32 = h128(c0.iter().zip(&c1).map(|(a, b)| a + b).collect())
                .into_iter().zip(&scales).map(|(x, s)| x * s).collect::<Vec<_>>();
            // Old path narrowed c0/c1 to BF16 before H128, then exchanged and zeroed rank 1.
            let s_old = h128(c0.iter().map(|&v| narrow(v)).zip(c1.iter().map(|&v| narrow(v)))
                .map(|(a, b)| a + b).collect())
                .into_iter().zip(&scales).map(|(x, s)| narrow(x * s)).collect::<Vec<_>>();
            // New path runs H128/svh per rank in F32 before narrowing to BF16 once.
            let s0_new = h128(c0).into_iter().zip(&scales).map(|(x, s)| narrow(x * s)).collect::<Vec<_>>();
            let s1_new = h128(c1).into_iter().zip(&scales).map(|(x, s)| narrow(x * s)).collect::<Vec<_>>();
            for n in 0..128 {
                let ref_val = r0[n] + r1[n] + s_fp32[n];
                let old_val = narrow(narrow(r0[n] + s_old[n]) + narrow(r1[n]));
                let new_val = narrow(narrow(r0[n] + s0_new[n]) + narrow(r1[n] + s1_new[n]));
                old_sq_err += f64::from(old_val - ref_val).powi(2);
                new_sq_err += f64::from(new_val - ref_val).powi(2);
            }
        }
        assert!(new_sq_err <= old_sq_err, "per-rank F32 H128/svh egress must not worsen RMS error");
        let source = include_str!("engine/native_dense.rs");
        let split = source.split("fn core_split(").nth(1).unwrap()
            .split("fn core_received(").next().unwrap();
        assert!(split.contains("if !layer.dense && key == \"down\""),
            "MoE shared_experts.down must never trigger pre-FFN core exchange");
        assert!(!source.contains("fn meet_shared_core(") && !source.contains("fn core_zero_shared("),
            "redundant MoE shared-core meet and rank-1 zeroing must be removed");
        let reduce = source.split("fn reduce_projection(").nth(1).unwrap()
            .split("/// The caller").next().unwrap();
        assert!(reduce.contains("if rank == 0 { (local, peer) } else { (peer, local) }"));
        assert!(reduce.contains(".reduced(&programs.add"));
        let producer = source.split("fn core_linear(").nth(1).unwrap()
            .split("pub(super) fn core_producer").next().unwrap();
        assert!(producer.contains("if Self::core_split(layer, key)"));
        assert!(producer.contains("projection.local_partial("));
        assert!(producer.contains("dense_trellis::retain_partial(&programs.narrow, core, retained,"));
        assert!(!producer.contains("copy_d2d_async"));
    }

    #[test]
    fn retained_row_views_keep_native_f32_k6_bf16_and_sum_distinct() {
        let source = include_str!("engine.rs");
        let view = source.split("impl<'a> StepRows<'a>").nth(1).unwrap()
            .split("pub(crate) struct GlmPeer").next().unwrap();
        // A common two-byte pitch would silently truncate the F32 sum view.
        assert!(view.contains("let width = arenas[plane].buffer.bytes / self.rows;"));
        assert!(view.contains("Ok([view(0)?, view(1)?, view(2)?])"));
        assert!(source.contains(".ranks[rank].attention_partial_bytes)?"));
        assert!(source.contains(".ranks[rank].down_partial_bytes)?, allocate(4)?])"));
        let native = include_str!("engine/native_dense.rs");
        let receive = native.split("fn core_received(").nth(1).unwrap()
            .split("pub(super) fn publish_projection").next().unwrap();
        assert!(receive.contains("let bytes = rows * self.cfg.hidden * element_bytes;"));
        let publish = native.split("pub(super) fn publish_projection").nth(1).unwrap()
            .split("fn reduce_projection(").next().unwrap();
        assert!(publish.contains("self.exchange()?.push_partial_to(rank, at, core.buffer, w.delta.buffer, dst.ptr, rows * self.cfg.hidden)"));
        let reduction = native.split("fn reduce_projection(").nth(1).unwrap()
            .split("pub(super) fn projection_norm").next().unwrap();
        assert!(reduction.contains("self.exchange()?.reduce_partial_narrow_on("));
        assert!(reduction.contains("if rank == 0 { (local, peer) } else { (peer, local) }"));
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
