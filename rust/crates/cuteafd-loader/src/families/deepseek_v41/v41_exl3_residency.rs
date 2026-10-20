//! Compressed layer residency matching B12x projection-native Trellis storage.
//! No weight dequantization or tier-wide temporary payload is required.
use crate::{OfficialV41Catalog, V41Exl3Manifest, V41Exl3ProjectionKind};
use anyhow::{ensure, Context, Result};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub enum V41Exl3Layer {
    Backbone(usize),
    Dspark(usize),
}

#[derive(Debug, Serialize)]
pub struct V41Exl3ResidentBuffer {
    pub name: String,
    /// Logical bytes; an empty projection still gets a 16-byte dummy allocation.
    pub bytes: usize,
    /// Initial native little-endian int32 words, also used for FP32 unit scales.
    pub initial_words: Vec<i32>,
}

#[derive(Debug, Serialize)]
pub struct V41Exl3Load {
    pub tensor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert: Option<usize>,
    pub bytes: usize,
    /// (Buffer index, byte offset). Rotations are read once and copied to each tier.
    pub destinations: Vec<(usize, usize)>,
}

#[derive(Debug, Serialize)]
pub struct V41Exl3Residency {
    pub layer: V41Exl3Layer,
    pub world: usize,
    pub rank: usize,
    pub intermediate_start: usize,
    pub experts: usize,
    pub intermediate: usize,
    pub tiers: Vec<usize>,
    /// Gate/up/down counts for each tier, passed unchanged to native launches.
    pub projection_counts: Vec<[usize; 3]>,
    /// Present only for compact runtime-width storage; legacy even splits stay unchanged.
    #[serde(skip)]
    pub unit_map: Option<cuteafd_core::tp_units::RankUnits>,
    pub buffers: Vec<V41Exl3ResidentBuffer>,
    pub loads: Vec<V41Exl3Load>,
}

impl V41Exl3Residency {
    /// Installed code follows the storage contract, not a directory-name guess.
    pub fn aot_layout(&self) -> String {
        let compact = if self.unit_map.is_some() { "-compact" } else { "" };
        format!("tp{}{compact}-rank{}", self.world, self.rank)
    }

    /// Pointer offsets and payload for the single CUDA arena used by loading.
    /// Rotation vectors are resident once per decoder tier; these buffers also
    /// include descriptors and unit maps absent from checkpoint tensor totals.
    pub fn device_arena_layout(&self) -> std::result::Result<(Vec<usize>, usize), crate::serving_capacity::CacheGeometryError> {
        use crate::serving_capacity::CacheGeometryError::Overflow;
        let mut bytes = 0usize;
        let mut offsets = Vec::with_capacity(self.buffers.len());
        for buffer in &self.buffers {
            bytes = bytes.checked_add(255).ok_or(Overflow("EXL3 arena alignment"))? & !255;
            offsets.push(bytes);
            bytes = bytes.checked_add(buffer.bytes.max(16)).ok_or(Overflow("EXL3 arena payload"))?;
        }
        let page = 2 * 1024 * 1024;
        bytes = bytes.checked_add(page - 1).ok_or(Overflow("EXL3 arena pages"))? / page * page;
        Ok((offsets, bytes))
    }

    /// Device payload budget, excluding allocator alignment and execution scratch.
    pub fn bytes(&self) -> usize {
        self.buffers.iter().map(|b| b.bytes.max(16)).sum()
    }

    pub fn staging_bytes(&self) -> usize {
        self.loads.iter().map(|l| l.bytes).max().unwrap_or(0)
    }

    pub fn scratch_bytes(&self, catalog: &OfficialV41Catalog) -> Result<usize> {
        self.loads.iter().try_fold(0, |n, load| {
            let scratch = if let Some(map) = &self.unit_map {
                catalog.exl3_tensor_gather(&load.tensor, map,
                    load.expert.context("compact EXL3 load lacks expert")?)?.scratch_bytes()
            } else {
                catalog.exl3_tensor_slice(&load.tensor, self.world, self.rank)?.scratch_bytes()
            };
            Ok(n.max(scratch))
        })
    }

    /// Read one independent job into caller-owned pinned staging. Compressed
    /// bytes retain their checkpoint layout; destinations can be uploaded on
    /// the caller's stream before reusing this staging slot.
    pub fn read_into(
        &self,
        catalog: &OfficialV41Catalog,
        index: usize,
        staging: &mut [u8],
        scratch: &mut [u8],
    ) -> Result<usize> {
        let load = self
            .loads
            .get(index)
            .context("EXL3 load index out of range")?;
        if let Some(prefix) = load.tensor.strip_suffix(".trellis") {
            let mut mcg = [0u8; 4];
            catalog.read_exl3_tensor_into(
                &format!("{prefix}.mcg"),
                self.world,
                self.rank,
                &mut mcg,
                &mut [],
            )?;
            ensure!(
                u32::from_le_bytes(mcg) == 0xcbac1fed,
                "unsupported EXL3 MCG multiplier for {prefix}"
            );
        }
        let bytes = if let Some(map) = &self.unit_map {
            ensure!(staging.len() >= load.bytes, "compact EXL3 staging too small");
            let bytes = catalog.read_exl3_tensor_units_into(&load.tensor, map,
                load.expert.context("compact EXL3 load lacks expert")?, staging, scratch)?;
            ensure!(bytes <= load.bytes, "compact EXL3 source exceeds admitted load");
            // Rotations retain the compiled maximum width, while compressed
            // weights have no padding. Zero scales make missing H128 blocks inert.
            staging[bytes..load.bytes].fill(0);
            load.bytes
        } else {
            catalog.read_exl3_tensor_into(&load.tensor, self.world, self.rank, staging, scratch)?
        };
        ensure!(bytes == load.bytes, "EXL3 residency/read size mismatch");
        if !load.tensor.ends_with(".trellis") {
            ensure!(
                staging[..bytes]
                    .chunks_exact(2)
                    .all(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7c00 != 0x7c00),
                "non-finite EXL3 rotation in {}",
                load.tensor
            );
        }
        Ok(bytes)
    }
}

impl V41Exl3Manifest {
    /// Plan whole-unit balanced owned-byte storage.
    /// Divisible geometry returns the legacy plan verbatim, including descriptors.
    pub fn balanced_residency(&self, layer: V41Exl3Layer, world: usize, rank: usize) -> Result<V41Exl3Residency> {
        use cuteafd_core::tp_units::RankUnits;
        let mut plan = self.residency(layer, world, rank)?;
        if self.experts.intermediate / 128 % world == 0 {
            return Ok(plan);
        }
        ensure!(self.decoder_tiers.len() == 2, "compact EXL3 requires a qualified two-tier export");
        let V41Exl3Layer::Backbone(index) = layer else {
            anyhow::bail!("compact EXL3 draft placement is not qualified");
        };
        let map = RankUnits::new(cuteafd_core::tp_units::balanced_rule(128, self.experts.intermediate / 128, world),
            plan.experts, self.experts.intermediate, 128, index, world, rank)?;
        let widths: Vec<_> = map.experts.iter().map(|units| units.len() * 128).collect();
        ensure!(widths.iter().all(|&width| width > 0), "compact EXL3 empty expert slice");
        let width = *widths.iter().max().context("compact EXL3 lacks experts")?;
        let stride = plan.experts * plan.tiers.len();
        let mut sizes = vec![[0usize; 3]; plan.tiers.len()];
        let mut offsets = vec![[0usize; 3]; plan.experts];
        for projection in 0..3 {
            for expert in 0..plan.experts {
                let kind = [V41Exl3ProjectionKind::Gate, V41Exl3ProjectionKind::Up,
                    V41Exl3ProjectionKind::Down][projection];
                let p = &self.projections[&self.naming.projection(false, index, expert, kind)];
                let tier = plan.tiers.binary_search(&p.bits).map_err(|_| anyhow::anyhow!("missing compact EXL3 tier"))?;
                offsets[expert][projection] = sizes[tier][projection];
                let bytes = self.experts.hidden.checked_mul(widths[expert])
                    .and_then(|n| n.checked_mul(p.bits)).context("compact EXL3 payload overflow")? / 8;
                sizes[tier][projection] = sizes[tier][projection].checked_add(bytes)
                    .context("compact EXL3 tier size overflow")?;
            }
        }
        let descriptor_bits = if plan.experts <= 512 { 9 } else { 10 };
        let descriptor_index = plan.buffers.iter().position(|b| b.name == "descriptor_map").unwrap();
        let mut descriptors = plan.buffers[descriptor_index].initial_words.clone();
        descriptors.resize(7 * stride, -1);
        for expert in 0..plan.experts {
            descriptors[3 * stride + expert] = i32::try_from(widths[expert])?;
            for projection in 0..3 {
                let tier = (descriptors[projection * stride + expert] as usize) >> descriptor_bits;
                if projection == 1 { offsets[expert][projection] += sizes[tier][0]; }
                descriptors[(4 + projection) * stride + expert] = i32::try_from(offsets[expert][projection] / 4)
                    .context("compact EXL3 offset exceeds int32 words")?;
            }
        }
        for (tier, size) in sizes.iter().enumerate() {
            plan.buffers.iter_mut().find(|b| b.name == format!("tier{tier}_w13")).unwrap().bytes = size[0] + size[1];
            plan.buffers.iter_mut().find(|b| b.name == format!("tier{tier}_w2")).unwrap().bytes = size[2];
        }
        plan.buffers[descriptor_index].bytes = descriptors.len() * 4;
        plan.buffers[descriptor_index].initial_words = descriptors;
        plan.buffers.iter_mut().find(|b| b.name == "intermediate_rotations").unwrap().bytes = stride * width * 3 * 2;
        for (expert, loads) in plan.loads.chunks_exact_mut(9).enumerate() {
            for (projection, loads) in loads.chunks_exact_mut(3).enumerate() {
                let kind = [V41Exl3ProjectionKind::Gate, V41Exl3ProjectionKind::Up,
                    V41Exl3ProjectionKind::Down][projection];
                let p = &self.projections[&self.naming.projection(false, index, expert, kind)];
                for load in loads {
                    load.expert = Some(expert);
                    if load.tensor.ends_with(".trellis") {
                        load.bytes = self.experts.hidden * widths[expert] * p.bits / 8;
                        load.destinations[0].1 = offsets[expert][projection];
                    } else {
                        let hidden = (projection < 2 && load.tensor.ends_with(".suh"))
                            || (projection == 2 && load.tensor.ends_with(".svh"));
                        if !hidden {
                            load.bytes = width * 2;
                            for (copy, destination) in load.destinations.iter_mut().enumerate() {
                                destination.1 = ((copy * plan.experts + expert) * 3 + projection) * width * 2;
                            }
                        }
                    }
                }
            }
        }
        plan.intermediate = width;
        plan.intermediate_start = 0;
        plan.unit_map = Some(map);
        Ok(plan)
    }

    pub fn residency(
        &self,
        layer: V41Exl3Layer,
        world: usize,
        rank: usize,
    ) -> Result<V41Exl3Residency> {
        let shape = self.experts;
        let (draft, index, experts) = match layer {
            V41Exl3Layer::Backbone(layer) => {
                ensure!(layer < shape.layers, "EXL3 backbone layer out of range");
                (false, layer, shape.experts)
            }
            V41Exl3Layer::Dspark(stage) => {
                ensure!(stage < shape.draft_stages, "EXL3 dSpark stage out of range");
                (true, stage, shape.draft_experts)
            }
        };
        ensure!(
            (1..=1024).contains(&experts),
            "EXL3 supports 1..1024 routed experts"
        );
        let mut projections = Vec::with_capacity(experts);
        for expert in 0..experts {
            let mut row = Vec::new();
            for kind in [
                V41Exl3ProjectionKind::Gate,
                V41Exl3ProjectionKind::Up,
                V41Exl3ProjectionKind::Down,
            ] {
                let p = self
                    .projections
                    .get(&self.naming.projection(draft, index, expert, kind))
                    .context("missing EXL3 resident projection")?;
                ensure!(
                    (2..=5).contains(&p.bits),
                    "EXL3 projection tier outside K2..K5"
                );
                row.push(p);
            }
            projections.push(row);
        }
        // All layers retain the package's checkpoint-wide descriptor tier IDs.
        let tiers = self.decoder_tiers.clone();
        let partition =
            projections[0][0].intermediate_partition(world, rank)?;
        let width = partition.end - partition.start;
        let hidden = shape.hidden;
        let mut counts = vec![[0; 3]; tiers.len()];
        let stride = experts * tiers.len();
        let descriptor_bits = if experts <= 512 { 9 } else { 10 };
        let mut descriptor = vec![-1; 3 * stride];
        let mut locals = vec![[0; 3]; experts];
        for (expert, row) in projections.iter().enumerate() {
            for (projection, p) in row.iter().enumerate() {
                ensure!(
                    p.intermediate_partition(world, rank)? == partition,
                    "inconsistent EXL3 intermediate partition"
                );
                let tier = tiers.binary_search(&p.bits).map_err(|_| {
                    anyhow::anyhow!("projection decoder absent from checkpoint family")
                })?;
                let local = counts[tier][projection];
                ensure!(
                    local < (1 << descriptor_bits),
                    "EXL3 descriptor local exceeds its width"
                );
                locals[expert][projection] = local;
                descriptor[projection * stride + expert] =
                    ((tier << descriptor_bits) | local) as i32;
                counts[tier][projection] += 1;
            }
        }
        let mut buffers = Vec::new();
        let mut add = |name: String, bytes: usize, initial_words: Vec<i32>| {
            let index = buffers.len();
            buffers.push(V41Exl3ResidentBuffer {
                name,
                bytes,
                initial_words,
            });
            index
        };
        let mut payload_buffers = Vec::new();
        for (tier, bit) in tiers.iter().enumerate() {
            let one = hidden * width * bit / 8;
            let w13 = add(
                format!("tier{tier}_w13"),
                (counts[tier][0] + counts[tier][1]) * one,
                vec![],
            );
            let w2 = add(format!("tier{tier}_w2"), counts[tier][2] * one, vec![]);
            payload_buffers.push([w13, w2]);
        }
        let gate = add("gate_suh".into(), stride * hidden * 2, vec![]);
        let up = add("up_suh".into(), stride * hidden * 2, vec![]);
        let down = add("down_svh".into(), stride * hidden * 2, vec![]);
        let intermediate = add(
            "intermediate_rotations".into(),
            stride * width * 3 * 2,
            vec![],
        );
        add("descriptor_map".into(), descriptor.len() * 4, descriptor);
        add(
            "global_to_combined".into(),
            experts * 4,
            (0..experts as i32).collect(),
        );
        add("dummy_scales".into(), 4, vec![0]);
        add("unit_scales".into(), experts * 4, vec![0x3f800000; experts]);
        let mut loads = Vec::new();
        for (expert, row) in projections.iter().enumerate() {
            for (projection, p) in row.iter().enumerate() {
                let tier = tiers.binary_search(&p.bits).map_err(|_| {
                    anyhow::anyhow!("projection decoder absent from checkpoint family")
                })?;
                let one = hidden * width * p.bits / 8;
                let slot =
                    locals[expert][projection] + if projection == 1 { counts[tier][0] } else { 0 };
                loads.push(V41Exl3Load {
                    tensor: format!("{}.trellis", p.name),
                    expert: None,
                    bytes: one,
                    destinations: vec![(
                        payload_buffers[tier][usize::from(projection == 2)],
                        slot * one,
                    )],
                });
                for suffix in ["suh", "svh"] {
                    let is_hidden =
                        (projection < 2 && suffix == "suh") || (projection == 2 && suffix == "svh");
                    let (buffer, bytes) = if is_hidden {
                        ([gate, up, down][projection], hidden * 2)
                    } else {
                        (intermediate, width * 2)
                    };
                    let destinations = (0..tiers.len())
                        .map(|copy| {
                            let global = copy * experts + expert;
                            (
                                buffer,
                                if is_hidden {
                                    global * bytes
                                } else {
                                    (global * 3 + projection) * bytes
                                },
                            )
                        })
                        .collect();
                    loads.push(V41Exl3Load {
                        tensor: format!("{}.{suffix}", p.name),
                        expert: None,
                        bytes,
                        destinations,
                    });
                }
            }
        }
        Ok(V41Exl3Residency {
            layer,
            world,
            rank,
            intermediate_start: partition.start,
            experts,
            intermediate: width,
            tiers,
            projection_counts: counts,
            unit_map: None,
            buffers,
            loads,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OfficialV41Config, V41Exl3Projection};
    use std::collections::BTreeMap;

    #[test]
    fn compact_owned_payloads_match_offsets_and_even_layouts_are_unchanged() -> Result<()> {
        for (hidden, intermediate, experts, tiers) in [
            (5120, 2304, 384, [3, 4]), (6144, 2048, 256, [4, 5]),
            (2560, 640, 512, [4, 5]),
        ] {
            for world in [2, 3, 4, 6].into_iter().filter(|&world| world <= intermediate / 128) {
                let manifest = fixture_geometry(&tiers, hidden, intermediate, experts);
                let units = intermediate / 128;
                let mut total_weights = 0;
                let mut resident_bytes = Vec::new();
                let mut allocated_bytes = Vec::new();
                for rank in 0..world {
                    let old = manifest.residency(V41Exl3Layer::Backbone(0), world, rank)?;
                    let plan = manifest.balanced_residency(V41Exl3Layer::Backbone(0), world,
                        rank)?;
                    if units % world == 0 {
                        assert_eq!(serde_json::to_value(&plan)?, serde_json::to_value(&old)?);
                        assert!(plan.unit_map.is_none());
                        assert_eq!(plan.aot_layout(), format!("tp{world}-rank{rank}"));
                    } else {
                        assert_eq!(plan.aot_layout(), format!("tp{world}-compact-rank{rank}"));
                        let map = plan.unit_map.as_ref().unwrap();
                        let stride = plan.experts * plan.tiers.len();
                        let descriptor = &plan.buffers.iter().find(|b| b.name == "descriptor_map").unwrap().initial_words;
                        assert_eq!(descriptor.len(), 7 * stride);
                        for (expert, loads) in plan.loads.chunks_exact(9).enumerate() {
                            assert_eq!(descriptor[3 * stride + expert] as usize, map.experts[expert].len() * 128);
                            for (projection, loads) in loads.chunks_exact(3).enumerate() {
                                let load = &loads[0];
                                assert_eq!(load.expert, Some(expert));
                                assert_eq!(load.destinations[0].1, descriptor[(4 + projection) * stride + expert] as usize * 4);
                                let tier = descriptor[projection * stride + expert] >> 9;
                                assert_eq!(load.bytes, hidden * map.experts[expert].len() * 128 * plan.tiers[tier as usize] / 8);
                            }
                        }
                    }
                    for (buffer_id, buffer) in plan.buffers.iter().enumerate().filter(|(_, b)|
                        b.name.ends_with("_w13") || b.name.ends_with("_w2")) {
                        let mut spans: Vec<_> = plan.loads.iter().flat_map(|load| load.destinations.iter()
                            .filter(move |(index, _)| *index == buffer_id)
                            .map(move |(_, offset)| (*offset, *offset + load.bytes))).collect();
                        spans.sort_unstable();
                        let mut end = 0;
                        for (start, next) in spans { assert_eq!(start, end); end = next; }
                        assert_eq!(end, buffer.bytes);
                        total_weights += buffer.bytes;
                    }
                    resident_bytes.push((old.bytes(), plan.bytes()));
                    allocated_bytes.push((old.device_arena_layout()?.1, plan.device_arena_layout()?.1));
                }
                let source_weights: usize = manifest.projections.values().map(|p| p.trellis_bytes()).sum();
                assert_eq!(total_weights, source_weights);
                if units % world != 0 {
                    // Balancing redistributes storage: small legacy ranks grow,
                    // while the maximum rank (the admission bottleneck) shrinks.
                    assert!(resident_bytes.iter().map(|(_, new)| new).max()
                        < resident_bytes.iter().map(|(old, _)| old).max());
                    assert!(allocated_bytes.iter().map(|(_, new)| new).max()
                        < allocated_bytes.iter().map(|(old, _)| old).max());
                } else {
                    assert!(allocated_bytes.iter().all(|(old, new)| old == new));
                }
                eprintln!("EXL3 H{hidden}/I{intermediate}/E{experts} TP{world} payload bytes old/compact: {resident_bytes:?}");
                eprintln!("EXL3 H{hidden}/I{intermediate}/E{experts} TP{world} allocated arena bytes old/compact: {allocated_bytes:?}");
            }
        }
        Ok(())
    }

    #[test]
    fn device_arena_preserves_pointer_alignment_and_bounds() {
        let mut plan = fixture(&[4, 5]).residency(V41Exl3Layer::Backbone(0), 1, 0).unwrap();
        let (offsets, bytes) = plan.device_arena_layout().unwrap();
        assert_eq!(bytes % (2 * 1024 * 1024), 0);
        let mut end = 0;
        for (&offset, buffer) in offsets.iter().zip(&plan.buffers) {
            assert_eq!(offset % 256, 0);
            assert!(offset >= end && offset - end < 256);
            end = offset + buffer.bytes.max(16);
        }
        assert!(bytes >= end && bytes - end < 2 * 1024 * 1024);
        plan.buffers[0].bytes = usize::MAX;
        assert!(plan.device_arena_layout().is_err());
    }

    fn fixture(tiers: &[usize]) -> V41Exl3Manifest {
        fixture_geometry(tiers, 5120, 2304, 384)
    }

    fn fixture_geometry(tiers: &[usize], hidden: usize, intermediate: usize, experts: usize) -> V41Exl3Manifest {
        let config = OfficialV41Config::from_json(include_bytes!("official-v41-config.json")).unwrap();
        let mut shape = crate::RoutedExpertShape::of_v41(&config);
        shape.hidden = hidden;
        shape.intermediate = intermediate;
        shape.experts = experts;
        let mut projections = BTreeMap::new();
        for expert in 0..experts {
            for (index, (name, kind)) in [
                ("w1", V41Exl3ProjectionKind::Gate),
                ("w3", V41Exl3ProjectionKind::Up),
                ("w2", V41Exl3ProjectionKind::Down),
            ]
            .into_iter()
            .enumerate()
            {
                let name = format!("layers.0.ffn.experts.{expert}.{name}");
                projections.insert(
                    name.clone(),
                    V41Exl3Projection {
                        name,
                        kind,
                        bits: tiers[(expert + index) % tiers.len()],
                        input_features: if index == 2 { intermediate } else { hidden },
                        output_features: if index == 2 { hidden } else { intermediate },
                    },
                );
            }
        }
        V41Exl3Manifest {
            experts: shape,
            config: Some(config),
            naming: crate::V41Exl3Naming::CheckpointNative,
            decoder_tiers: crate::families::deepseek_v41::v41_exl3::decoder_family(&projections).unwrap(),
            projections,
            ple_quantization: None,
            mtp_experts: crate::families::deepseek_v41::v41_exl3::V41Exl3MtpExperts::Exl3,
            storage: crate::formats::exl3_storage::Exl3StorageSource::GptqModel,
        }
    }

    #[test]
    fn kimi_descriptors_preserve_high_expert_ids_in_both_tiers() {
        for tier in 0..2 {
            let mut manifest = fixture(&[2, 3]);
            manifest.experts.experts = 896;
            manifest.experts.hidden = 3584;
            manifest.experts.intermediate = 3072;
            manifest.experts.topk = 16;
            manifest.projections.clear();
            for expert in 0..896 {
                for (suffix, kind) in [
                    ("w1", V41Exl3ProjectionKind::Gate),
                    ("w3", V41Exl3ProjectionKind::Up),
                    ("w2", V41Exl3ProjectionKind::Down),
                ] {
                    let name = format!("layers.0.ffn.experts.{expert}.{suffix}");
                    let down = suffix == "w2";
                    manifest.projections.insert(
                        name.clone(),
                        V41Exl3Projection {
                            name,
                            kind,
                            bits: 2 + tier,
                            input_features: if down { 3072 } else { 3584 },
                            output_features: if down { 3584 } else { 3072 },
                        },
                    );
                }
            }
            for (world, rank) in [(1, 0), (6, 0), (6, 5)] {
                let plan = manifest
                    .residency(V41Exl3Layer::Backbone(0), world, rank)
                    .unwrap();
                assert_eq!(plan.intermediate, 3072 / world);
                let descriptor = &plan
                    .buffers
                    .iter()
                    .find(|b| b.name == "descriptor_map")
                    .unwrap()
                    .initial_words;
                for projection in 0..3 {
                    for expert in [0, 511, 512, 895] {
                        assert_eq!(
                            descriptor[projection * 1792 + expert],
                            ((tier << 10) | expert) as i32
                        );
                    }
                    assert!(descriptor[projection * 1792 + 896..(projection + 1) * 1792]
                        .iter()
                        .all(|&v| v == -1));
                }
            }
        }
    }

    #[test]
    fn mixed_and_uniform_payload_destinations_cover_buffers_without_overlap() {
        for tiers in [&[2][..], &[3], &[4], &[5], &[3, 4], &[2, 3, 4, 5]] {
            let manifest = fixture(tiers);
            for world in [1, 2, 3, 4] {
                let mut total_width = 0;
                for rank in 0..world {
                    let plan = manifest
                        .residency(V41Exl3Layer::Backbone(0), world, rank)
                        .unwrap();
                    total_width += plan.intermediate;
                    let mut spans = vec![Vec::new(); plan.buffers.len()];
                    for job in &plan.loads {
                        for &(buffer, offset) in &job.destinations {
                            assert_eq!(offset % 16, 0);
                            spans[buffer].push(offset..offset + job.bytes);
                        }
                    }
                    for (buffer, spans) in plan.buffers.iter().zip(&mut spans) {
                        spans.sort_by_key(|span| span.start);
                        let mut cursor = buffer.initial_words.len() * 4;
                        for span in spans {
                            assert_eq!(span.start, cursor, "{}", buffer.name);
                            cursor = span.end;
                        }
                        assert_eq!(cursor, buffer.bytes, "{}", buffer.name);
                    }
                    assert_eq!(plan.loads.len(), 384 * 9);
                    for projection in 0..3 {
                        assert_eq!(
                            plan.projection_counts
                                .iter()
                                .map(|c| c[projection])
                                .sum::<usize>(),
                            384
                        );
                    }
                    let payload: usize = plan
                        .buffers
                        .iter()
                        .filter(|b| b.name.starts_with("tier"))
                        .map(|b| b.bytes)
                        .sum();
                    let expected: usize = manifest
                        .projections
                        .values()
                        .map(|p| 5120 * plan.intermediate * p.bits / 8)
                        .sum();
                    assert_eq!(payload, expected);
                }
                assert_eq!(total_width, 2304);
            }
        }
    }

    #[test]
    fn missing_layer_and_bad_partitions_are_rejected() {
        let manifest = fixture(&[3, 4]);
        assert!(manifest
            .residency(V41Exl3Layer::Backbone(40), 1, 0)
            .is_err());
        assert!(manifest.residency(V41Exl3Layer::Dspark(3), 1, 0).is_err());
        // TP3 and TP6 are exact whole-block splits; reject invalid ranks/worlds.
        assert!(manifest.residency(V41Exl3Layer::Backbone(0), 3, 3).is_err());
        assert!(manifest.residency(V41Exl3Layer::Backbone(0), 6, 6).is_err());
        assert!(manifest.residency(V41Exl3Layer::Backbone(0), 9, 0).is_err());
        assert!(manifest.residency(V41Exl3Layer::Backbone(0), 4, 4).is_err());
    }

    /// Whole-block TP3/TP6 shards tile 2304 channels without TP4's unequal split.
    #[test]
    fn tp3_tp6_residency_is_disjoint_and_equal() {
        for tiers in [&[2, 3][..], &[3, 4]] {
            let manifest = fixture(tiers);
            let full = manifest
                .residency(V41Exl3Layer::Backbone(0), 1, 0)
                .unwrap();
            let full_payload: usize = full
                .buffers
                .iter()
                .filter(|b| b.name.starts_with("tier"))
                .map(|b| b.bytes)
                .sum();
            for world in [3, 6] {
                let width = 2304 / world;
                let shard_payload: Vec<usize> = (0..world)
                    .map(|rank| {
                        let plan = manifest
                            .residency(
                                V41Exl3Layer::Backbone(0),
                                world,
                                rank,
                            )
                            .unwrap();
                        assert_eq!(
                            (plan.world, plan.rank),
                            (world, rank)
                        );
                        assert_eq!(plan.intermediate, width);
                        assert_eq!(plan.intermediate_start, rank * width);
                        assert_eq!(plan.tiers, tiers);
                        plan.buffers
                            .iter()
                            .filter(|b| b.name.starts_with("tier"))
                            .map(|b| b.bytes)
                            .sum()
                    })
                    .collect();
                assert!(shard_payload.iter().all(|&bytes| bytes == shard_payload[0]));
                assert_eq!(shard_payload.iter().sum::<usize>(), full_payload);
            }
        }
    }
    #[test]
    fn layer_subsets_keep_checkpoint_family_and_exact_payloads() {
        for family in [&[3, 4][..], &[2, 5], &[2, 3, 4, 5]] {
            let mut manifest = fixture(&family[..1]);
            let original: Vec<_> = manifest.projections.values().cloned().collect();
            for (layer, &bits) in family.iter().enumerate().skip(1) {
                for mut p in original.clone() {
                    p.name = p.name.replacen("layers.0.", &format!("layers.{layer}."), 1);
                    p.bits = bits;
                    manifest.projections.insert(p.name.clone(), p);
                }
            }
            manifest.decoder_tiers =
                crate::families::deepseek_v41::v41_exl3::decoder_family(&manifest.projections).unwrap();
            for world in [1, 2, 3, 4] {
                for rank in 0..world {
                    for (layer, &bits) in family.iter().enumerate() {
                        let plan = manifest
                            .residency(V41Exl3Layer::Backbone(layer), world, rank)
                            .unwrap();
                        assert_eq!(plan.tiers, family);
                        for (tier, counts) in plan.projection_counts.iter().enumerate() {
                            assert_eq!(*counts, [if tier == layer { 384 } else { 0 }; 3]);
                        }
                        let payload: usize = plan
                            .buffers
                            .iter()
                            .filter(|b| b.name.starts_with("tier"))
                            .map(|b| b.bytes)
                            .sum();
                        assert_eq!(payload, 384 * 3 * 5120 * plan.intermediate * bits / 8);
                        let descriptor = plan
                            .buffers
                            .iter()
                            .find(|b| b.name == "descriptor_map")
                            .unwrap();
                        for projection in 0..3 {
                            for expert in 0..384 {
                                assert_eq!(
                                    descriptor.initial_words
                                        [projection * 384 * family.len() + expert],
                                    ((layer << 9) | expert) as i32
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
