//! Exact aligned regions used by the coordinator's physical weight arenas.
use anyhow::{ensure, Context, Result};
use std::{collections::BTreeMap, ops::Range};

/// One physical CUDA request and its immutable tensor views. IDs survive
/// planning/loading so granularity owners charge backings, not views.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightArenaLayout {
    id: String,
    bytes: usize,
    views: BTreeMap<String, WeightArenaView>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightArenaView { offset: usize, bytes: usize }
impl WeightArenaView {
    pub fn offset(&self) -> usize { self.offset }
    pub fn bytes(&self) -> usize { self.bytes }
}

impl WeightArenaLayout {
    /// Preserve cudaMalloc's 256-byte alignment for every consumer (including
    /// the native K6 dispatcher and TMA's 16-byte tensor-map bases).
    pub fn new(id: impl Into<String>, tensors: impl IntoIterator<Item = (String, usize)>) -> Result<Self> {
        let id = id.into();
        ensure!(!id.is_empty(), "empty weight arena ID");
        let tensors = tensors.into_iter().collect::<Vec<_>>();
        ensure!(!tensors.is_empty(), "empty weight arena scope");
        ensure!(tensors.iter().all(|(name, bytes)| !name.is_empty() && *bytes > 0), "empty weight arena tensor");
        let packed = PackedWeightRegions::new(tensors.iter().map(|(_, bytes)| *bytes))?;
        let mut views = BTreeMap::new();
        for ((name, bytes), region) in tensors.into_iter().zip(packed.regions) {
            ensure!(views.insert(name, WeightArenaView { offset: region.start, bytes }).is_none(),
                "duplicate weight arena tensor");
        }
        Ok(Self { id, bytes: packed.bytes, views })
    }
    pub fn id(&self) -> &str { &self.id }
    pub fn bytes(&self) -> usize { self.bytes }
    pub fn views(&self) -> &BTreeMap<String, WeightArenaView> { &self.views }
    pub fn view(&self, name: &str) -> Option<WeightArenaView> { self.views.get(name).copied() }
    pub fn payload_bytes(&self) -> usize { self.views.values().map(WeightArenaView::bytes).sum() }
    pub fn padding_bytes(&self) -> usize { self.bytes - self.payload_bytes() }
}

#[derive(Debug, Clone)]
pub struct PackedWeightRegions {
    pub regions: Vec<Range<usize>>,
    pub bytes: usize,
}

impl PackedWeightRegions {
    pub fn new(sizes: impl IntoIterator<Item = usize>) -> Result<Self> {
        let mut regions = Vec::new();
        let mut bytes = 0usize;
        for size in sizes {
            bytes = bytes.checked_add(255).context("weight region alignment overflow")? & !255;
            let end = bytes.checked_add(size).context("weight region extent overflow")?;
            regions.push(bytes..end);
            bytes = end;
        }
        Ok(Self { regions, bytes })
    }
}

/// Final weight and scale views shared by package admission and runtime mounts.
pub fn fp8_expert_regions(tensors: &crate::formats::fp8_experts::Fp8ExpertTensors,
    layers: &[usize], tp: usize, rank: usize,
    slicing: crate::formats::fp8_experts::Slicing) -> Result<Vec<(String, usize)>> {
    use crate::formats::fp8_experts::Fp8Projection;
    let mut regions = Vec::new();
    for &layer in layers {
        let tensors = tensors.for_layer_metadata(layer)?;
        for (index, projection) in Fp8Projection::ALL.into_iter().enumerate() {
            let (weights, _) = tensors.slice_bytes_with(projection, tp, rank, slicing)?;
            for (plane, bytes) in [(index * 2, weights * tensors.shape().experts),
                (index * 2 + 1, tensors.scale_region_bytes_with(projection, tp, rank, slicing)?)] {
                if bytes > 0 { regions.push((format!("{layer}.{plane}"), bytes)); }
            }
        }
    }
    Ok(regions)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn package_formats_selected_tp1_tp2_planes_match_load_extent() -> Result<()> {
        use crate::formats::fp8_experts::{ExpertFormat, Slicing};
        use crate::plan::testing::{fp8, mimo_flash_config, mxfp4, nvfp4, t, write_snapshot};
        for format in [ExpertFormat::Fp8Block128, ExpertFormat::Mxfp4, ExpertFormat::Nvfp4] {
            for bf16_scales in [false, true] {
                let root = tempfile::tempdir()?;
                let mut config = mimo_flash_config();
                config["hidden_size"] = 128.into();
                config["n_routed_experts"] = 2.into();
                config["num_experts_per_tok"] = 2.into();
                config["moe_intermediate_size"] = 1024.into();
                config["num_hidden_layers"] = 3.into();
                config["hybrid_layer_pattern"] = serde_json::json!([0, 1, 1]);
                config["moe_layer_freq"] = serde_json::json!([0, 1, 1]);
                if format == ExpertFormat::Nvfp4 {
                    config["quantization_config"]["producer"] = serde_json::json!({"name": "modelopt"});
                    config["quantization_config"]["config_groups"] = serde_json::json!({"group_0": {
                        "targets": ["Linear"],
                        "weights": {"dynamic": false, "group_size": 16, "num_bits": 4, "type": "float"},
                        "input_activations": {"dynamic": false, "group_size": 16, "num_bits": 4, "type": "float"}
                    }});
                }
                let mut headers = Vec::new();
                for expert in 0..2 {
                    for (projection, rows, columns) in [("gate", 1024, 128), ("up", 1024, 128), ("down", 128, 1024)] {
                        let name = format!("model.layers.1.mlp.experts.{expert}.{projection}_proj");
                        headers.extend(match format {
                            ExpertFormat::Nvfp4 => nvfp4(&name, rows, columns),
                            ExpertFormat::Mxfp4 => mxfp4(&name, rows, columns),
                            _ => fp8(&name, rows, columns, None),
                        });
                        headers.push(t(format!("model.layers.2.mlp.experts.{expert}.{projection}_proj.weight"),
                            "BF16", &[rows, columns]));
                    }
                }
                if bf16_scales {
                    for (name, dtype, _) in &mut headers {
                        if name.ends_with("_scale_inv") { *dtype = "BF16"; }
                    }
                }
                write_snapshot(root.path(), &config, &headers, None);
                let catalog = crate::read_expert_catalog(root.path())?;
                let tensors = catalog.fp8().context("package inventory missing")?;
                for world in [1, 2] {
                    for rank in 0..world {
                        for slicing in [Slicing::Padded, Slicing::Blocks(128)] {
                            let regions = fp8_expert_regions(tensors, &[1, 2], world, rank, slicing)?;
                            let layout = WeightArenaLayout::new("fixture", regions.clone())?;
                            for (name, bytes) in &regions {
                                let view = layout.view(name).unwrap();
                                assert_eq!(view.offset() % 256, 0);
                                assert_eq!(view.bytes(), *bytes);
                            }
                            for layer in [1, 2] {
                                let expected = tensors.for_layer_metadata(layer)?.layer_load(layer, world, rank, slicing)?.resident;
                                assert_eq!(regions.iter().filter(|(name, _)| name.starts_with(&format!("{layer}.")))
                                    .map(|(_, bytes)| bytes).sum::<usize>(), expected);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn preserves_plane_order_alignment_and_final_extent() {
        let layout = PackedWeightRegions::new([3, 0, 257, 16]).unwrap();
        assert_eq!(layout.regions, [0..3, 256..256, 256..513, 768..784]);
        assert_eq!(layout.bytes, 784);
        assert_eq!(PackedWeightRegions::new([]).unwrap().bytes, 0);
        assert!(PackedWeightRegions::new([usize::MAX, 1]).is_err());
        assert!(PackedWeightRegions::new([256, usize::MAX]).is_err());
    }

    #[test]
    fn named_arena_views_match_the_single_physical_request() -> Result<()> {
        let layout = WeightArenaLayout::new("core/gpu0", [("payload".into(), 257), ("scale".into(), 2)])?;
        assert_eq!(layout.id(), "core/gpu0");
        assert_eq!(layout.bytes(), 514);
        assert_eq!(layout.payload_bytes(), 259);
        assert_eq!(layout.padding_bytes(), 255);
        assert_eq!(layout.view("scale").unwrap().offset(), 512);
        assert_eq!(layout.view("scale").unwrap().bytes(), 2);
        assert!(layout.view("absent").is_none());
        let mut regions = layout.views().values().copied().collect::<Vec<_>>();
        regions.sort_by_key(WeightArenaView::offset);
        let mut end = 0;
        for region in regions {
            assert_eq!(region.offset() % 256, 0);
            assert!(region.offset() >= end);
            end = region.offset() + region.bytes();
        }
        assert_eq!(end, layout.bytes());
        assert!(WeightArenaLayout::new("x", []).is_err());
        assert!(WeightArenaLayout::new("", [("a".into(), 1)]).is_err());
        assert!(WeightArenaLayout::new("x", [("".into(), 1)]).is_err());
        assert!(WeightArenaLayout::new("x", [("same".into(), 1), ("same".into(), 2)]).is_err());
        assert!(WeightArenaLayout::new("x", [("empty".into(), 0)]).is_err());
        assert!(WeightArenaLayout::new("x", [("a".into(), usize::MAX), ("b".into(), 1)]).is_err());
        Ok(())
    }
}
