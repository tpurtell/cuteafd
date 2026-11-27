//! Header-native Hy4-flash ViT on the generic still tower: biased patch-16
//! raster embedding, scale-factor learned positions, separate biased Q/K/V,
//! exact GELU, RMSNorm + conv merger and checkpoint newline rows.
use super::*;
use anyhow::Context;
use cuteafd_ffi::vision::{TowerExtension, TowerNorm, TowerProjector};
use cuteafd_loader::families::hy4_flash::vision::{VisionContract, MAX_MERGED_ROWS, POSITION_EDGE};

impl TowerSpec {
    fn hy4_tower(
        snapshot: &Path,
        max_output_rows: usize,
        family: cuteafd_loader::media::ImageFamily,
        contract: VisionContract,
        packed: cuteafd_loader::families::hy4_flash::vision::PackedVision,
        catalog: BTreeMap<String, (PathBuf, cuteafd_loader::SafetensorsTensorMetadata)>,
    ) -> anyhow::Result<Self> {
        let mut reads = Vec::with_capacity(packed.tensors.len());
        for tensor in &packed.tensors {
            let (path, metadata) = catalog.get(&tensor.metadata.name).context("Hy4 tower catalog changed")?;
            reads.push(TensorRead { path: path.clone(), metadata: metadata.clone(),
                destination: usize::try_from(tensor.destination)?, transform: ReadTransform::Direct });
        }
        let at = |name: &str| packed.offset(name);
        let (h, o, m) = (contract.hidden as u32, contract.output as u32, contract.merger_hidden as u32);
        let extension = TowerExtension {
            block_norm_kind: 1,
            activation: 1,
            block_norm_epsilon: contract.eps,
            attention_width: h,
            temporal_patch: 1,
            pixel_layout: 1,
            temporal_input: 0,
            rotary_kind: 0,
            passthrough_dims: (contract.hidden / contract.heads) as u32,
            absolute_kind: 2,
            position_height: POSITION_EDGE as u32,
            position_width: POSITION_EDGE as u32,
            interpolation: 2,
            absolute_position: at("vit.embeddings.position_embedding.weight")?,
            final_norm: TowerNorm { kind: 2, epsilon: contract.eps, weight: at("vit.perceive.norm.weight")?, bias: NO_VISION_OFFSET },
            spatial_merge: 2,
            projection_order: 3,
            projectors: [TowerProjector {
                kind: 1, activation: 1, input_width: 4 * h, hidden_width: m, output_width: o,
                fc1: at("vit.perceive.proj.0.weight")?, fc1_bias: at("vit.perceive.proj.0.bias")?,
                fc2: at("vit.perceive.proj.2.weight")?, fc2_bias: at("vit.perceive.proj.2.bias")?,
                ..Default::default()
            }, Default::default()],
            ..Default::default()
        };
        let mut merger_extra = [NO_VISION_OFFSET; 8];
        merger_extra[0] = at("vit.perceive.image_newline")?;
        let mut native = VisionSpec {
            abi_version: 4,
            max_tokens: u32::try_from(max_output_rows)?,
            output_width: o,
            weight_bytes: packed.arena_bytes,
            patch: at("vit.embeddings.patch_embedding.weight")?,
            patch_bias: at("vit.embeddings.patch_embedding.bias")?,
            hidden: h,
            heads: contract.heads as u32,
            kv_heads: contract.heads as u32,
            head_dim: (contract.hidden / contract.heads) as u32,
            intermediate: contract.intermediate as u32,
            patch_size: 16,
            merger_width: 4 * h,
            norm_eps: contract.eps,
            pos_embed: NO_VISION_OFFSET,
            inv_freq: NO_VISION_OFFSET,
            merger_norm: NO_VISION_OFFSET,
            merger_norm_bias: NO_VISION_OFFSET,
            merger_fc1: NO_VISION_OFFSET,
            merger_fc1_bias: NO_VISION_OFFSET,
            merger_fc2: NO_VISION_OFFSET,
            merger_fc2_bias: NO_VISION_OFFSET,
            merger_extra,
            tower: Some(extension),
            ..Default::default()
        }.with_depth(contract.layers);
        for layer in 0..contract.layers {
            let field = |suffix: &str| at(&format!("vit.layers.{layer}.{suffix}"));
            // Q/K/V weights and biases are adjacent (checked by the pack); the
            // native tower projects them separately from these base offsets.
            native.blocks[layer] = VisionBlock {
                qkv: field("self_attn.q_proj.weight")?,
                qkv_bias: field("self_attn.q_proj.bias")?,
                proj: field("self_attn.o_proj.weight")?,
                proj_bias: field("self_attn.o_proj.bias")?,
                gate_up: field("mlp.dense_h_to_4h.weight")?,
                gate_up_bias: field("mlp.dense_h_to_4h.bias")?,
                down: field("mlp.dense_4h_to_h.weight")?,
                down_bias: field("mlp.dense_4h_to_h.bias")?,
                norm1: field("input_layernorm.weight")?,
                norm2: field("post_attention_layernorm.weight")?,
                key0_bias: NO_VISION_OFFSET,
                window: 0,
                column_order: 0,
            };
            native.norm1_bias[layer] = field("input_layernorm.bias")?;
            native.norm2_bias[layer] = field("post_attention_layernorm.bias")?;
            native.q_norm[layer] = NO_VISION_OFFSET;
            native.k_norm[layer] = NO_VISION_OFFSET;
        }
        let _ = snapshot;
        Ok(Self::from_parts(native, reads)?.with_family(family))
    }

    pub fn hy4_flash(snapshot: &Path, max_output_rows: usize) -> anyhow::Result<Self> {
        anyhow::ensure!((1..=MAX_MERGED_ROWS).contains(&max_output_rows),
            "Hy4-flash vision merged capacity must be 1..={MAX_MERGED_ROWS}");
        let cfg = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
        let contract = VisionContract::from_config(&cfg)?;
        let catalog = tower_catalog(snapshot, "vit.")?;
        let packed = contract.pack(catalog.values().map(|(_, metadata)| metadata))?;
        Self::hy4_tower(
            snapshot,
            max_output_rows,
            cuteafd_loader::media::ImageFamily::Hy4Flash,
            contract,
            packed,
            catalog,
        )
    }

    pub fn hy4(snapshot: &Path, max_output_rows: usize) -> anyhow::Result<Self> {
        anyhow::ensure!((1..=MAX_MERGED_ROWS).contains(&max_output_rows),
            "Hy4 vision merged capacity must be 1..={MAX_MERGED_ROWS}");
        let cfg = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
        let contract = cuteafd_loader::families::hy4::vision::vision_contract(&cfg)?;
        let catalog = tower_catalog(snapshot, "vit.")?;
        let packed = cuteafd_loader::families::hy4::vision::pack(
            &contract,
            catalog.values().map(|(_, metadata)| metadata),
        )?;
        Self::hy4_tower(
            snapshot,
            max_output_rows,
            cuteafd_loader::media::ImageFamily::Hy4,
            contract,
            packed,
            catalog,
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    /// Header-only Hy4 snapshot: real config, synthetic sparse `vit.*` shard.
    pub(crate) fn snapshot() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let cfg: serde_json::Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/families/hy4_flash/fixtures/vision_config.json"))).unwrap();
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
        let contract = VisionContract::from_config(&cfg).unwrap();
        let mut header = serde_json::Map::new();
        let mut end = 0u64;
        for (name, shape) in contract.tensor_shapes() {
            let start = end;
            end += shape.iter().product::<usize>() as u64 * 2;
            header.insert(name, serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[start,end]}));
        }
        let bytes = serde_json::to_vec(&header).unwrap();
        let mut f = File::create(dir.path().join("vit.safetensors")).unwrap();
        f.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&bytes).unwrap();
        f.set_len(8 + bytes.len() as u64 + end).unwrap();
        let mut map = serde_json::Map::new();
        for name in header.keys() { map.insert(name.clone(), "vit.safetensors".into()); }
        map.insert("model.layers.0.unread.weight".into(), "absent-lm.safetensors".into());
        std::fs::write(dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": map})).unwrap()).unwrap();
        dir
    }

    #[test]
    fn hy4_headers_build_a_valid_generic_tower_without_lm_reads() {
        let dir = snapshot();
        let spec = TowerSpec::from_snapshot(dir.path(), MAX_MERGED_ROWS).unwrap();
        assert_eq!(spec.image_family(), cuteafd_loader::media::ImageFamily::Hy4Flash);
        assert_eq!(spec.reads.len(), 441);
        assert!(spec.reads.iter().all(|r| r.destination % 256 == 0 && r.transform == ReadTransform::Direct));
        let tower = spec.native.tower.unwrap();
        assert_eq!((tower.absolute_kind, tower.projection_order, spec.native.head_dim), (2, 3, 72));
        assert_ne!(spec.native.merger_extra[0], NO_VISION_OFFSET);
        assert_eq!(spec.row_markers(), 1);
        assert!(TowerSpec::hy4_flash(dir.path(), MAX_MERGED_ROWS + 1).is_err());
        let mut wrong = spec.native.clone();
        wrong.tower.as_mut().unwrap().projection_order = 1;
        assert!(wrong.tower.unwrap().validate(&wrong).is_err());
        let mut missing = spec.native.clone();
        missing.merger_extra[0] = NO_VISION_OFFSET;
        assert!(missing.tower.unwrap().validate(&missing).is_err());
        assert_ne!(spec.encoder_id("r", 120), spec.encoder_id("r", 121));
    }

    /// Host-only (no CUDA): the native validator admits the real Hy4 spec.
    #[test]
    #[ignore = "requires VISION_SNAPSHOT and VISION_LIBRARY (no CUDA initialization)"]
    fn hy4_native_validator_admits_the_real_tower() {
        let snapshot = std::path::PathBuf::from(std::env::var("VISION_SNAPSHOT").unwrap());
        let library = std::path::PathBuf::from(std::env::var("VISION_LIBRARY").unwrap());
        let spec = TowerSpec::hy4_flash(&snapshot, MAX_MERGED_ROWS).unwrap();
        let ledger = NativeVision::required(&library, &spec.native).unwrap();
        assert_eq!(ledger.weights, spec.native.weight_bytes);
        println!("hy4 tower ledger {ledger:?}");
    }

    fn read_f32(path: &Path) -> Vec<f32> {
        std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
            .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    }
    fn bf16(bits: u16) -> f32 { f32::from_bits(u32::from(bits) << 16) }
    /// (cosine, max-abs, relative L2) of engine rows against the oracle stage.
    fn compare(actual: &[f32], expected: &[f32]) -> (f64, f64, f64) {
        assert_eq!(actual.len(), expected.len(), "stage extent differs");
        let (mut dot, mut aa, mut ee, mut diff, mut worst) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (&a, &e) in actual.iter().zip(expected) {
            let (a, e) = (f64::from(a), f64::from(e));
            dot += a * e; aa += a * a; ee += e * e; diff += (a - e) * (a - e);
            worst = worst.max((a - e).abs());
        }
        (dot / (aa.sqrt() * ee.sqrt()).max(f64::MIN_POSITIVE), worst, (diff / ee.max(f64::MIN_POSITIVE)).sqrt())
    }

    /// Hardware (coordinator dev container): the real `vit.*` tower against the
    /// independent serving-mode oracle, per stage, per fixture, single and batched.
    /// `HY4_VISION_GOLDEN` holds `vision/<fixture>/{rgb.u8,meta.json,stage_*.f32}`;
    /// writes `HY4_VISION_REPORT` (JSON) when set.
    #[test]
    #[ignore = "requires VISION_SNAPSHOT, VISION_LIBRARY, HY4_VISION_GOLDEN and a SM120/SM121 GPU"]
    fn hy4_tower_matches_the_serving_oracle_per_stage() {
        let snapshot = std::path::PathBuf::from(std::env::var("VISION_SNAPSHOT").unwrap());
        let library = std::path::PathBuf::from(std::env::var("VISION_LIBRARY").unwrap());
        let golden = std::path::PathBuf::from(std::env::var("HY4_VISION_GOLDEN").unwrap()).join("vision");
        let processor = cuteafd_loader::media::ProcessorConfig::from_snapshot(&snapshot,
            cuteafd_loader::media::ImageFamily::Hy4Flash).unwrap();
        let lut = normalization_lut(&processor);
        let spec = TowerSpec::hy4_flash(&snapshot, MAX_MERGED_ROWS).unwrap();
        let depth = spec.native.depth as i32;
        let required = NativeVision::required(&library, &spec.native).unwrap();
        let mut runtime = VitRuntime::load(spec, &library, 0, required.total_bytes()).unwrap();
        // SAFETY: the same trusted native library the tower owner loaded.
        let native = unsafe { cuteafd_ffi::NativeLibrary::load(&library) }.unwrap();
        // Per-image fixtures carry `grid`; `batch/` is the reference's own batched turn.
        let meta_of = |p: &Path| -> Option<serde_json::Value> { serde_json::from_slice(&std::fs::read(p.join("meta.json")).ok()?).ok() };
        let mut fixtures: Vec<_> = std::fs::read_dir(&golden).unwrap().map(|e| e.unwrap().path())
            .filter(|p| meta_of(p).is_some_and(|m| m.get("grid").is_some())).collect();
        fixtures.sort();
        assert!(!fixtures.is_empty(), "no Hy4 vision fixtures under {}", golden.display());
        let mut report = serde_json::Map::new();
        let mut batch_rgb = Vec::new();
        let mut batch_grids = Vec::new();
        let mut singles = Vec::new();
        let (mut worst_cos, mut worst_out) = (1f64, 0f64);
        for fixture in &fixtures {
            let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture.join("meta.json")).unwrap()).unwrap();
            let grid: [usize; 2] = serde_json::from_value(meta["grid"].clone()).unwrap();
            let rgb = std::fs::read(fixture.join("rgb.u8")).unwrap();
            let rows = cuteafd_ffi::vision::output_rows(grid, true);
            let mut output = vec![0u16; rows * 3072];
            let mut stages: Vec<(String, Vec<f32>)> = Vec::new();
            runtime.native.encode_batch_observed(&rgb, &[grid], &lut, &mut output, &mut |stage, data, rows, width| {
                let extent = rows * width * 2;
                let mut bytes = vec![0u8; extent];
                native.copy_d2h_quiescent(&mut bytes, cuteafd_ffi::CuteafdDeviceBuffer { ptr: data as *mut _, bytes: extent,
                    device_id: 0, flags: 0 }).map_err(|e| cuteafd_ffi::vision::VisionError::Runtime(e.to_string()))?;
                let name = match stage { 0 => "0".to_string(), -1 => "pos".into(), s if s == depth + 1 => "28".into(),
                    s if s == depth + 2 => "out".into(), s => s.to_string() };
                stages.push((name, bytes.chunks_exact(2).map(|b| bf16(u16::from_le_bytes([b[0], b[1]]))).collect()));
                Ok(())
            }).unwrap();
            let mut entry = serde_json::Map::new();
            for (name, actual) in &stages {
                let path = fixture.join(format!("stage_{name}.f32"));
                if !path.is_file() { continue; }
                let (cos, max_abs, rel) = compare(actual, &read_f32(&path));
                if name == "out" { worst_out = worst_out.max(rel); }
                worst_cos = worst_cos.min(cos);
                entry.insert(name.clone(), serde_json::json!({"cosine": cos, "max_abs": max_abs, "rel_l2": rel}));
            }
            report.insert(fixture.file_name().unwrap().to_string_lossy().into(), entry.into());
            batch_rgb.extend_from_slice(&rgb);
            batch_grids.push(grid);
            singles.extend(output);
        }
        // Batched owner turn must equal the single-image encodes exactly.
        let mut batched = vec![0u16; singles.len()];
        runtime.native.encode_batch_into(&batch_rgb, &batch_grids, &lut, &mut batched).unwrap();
        let identical = batched == singles;
        report.insert("batch_equals_singles".into(), identical.into());
        if let Some(meta) = meta_of(&golden.join("batch")) {
            let order: Vec<String> = serde_json::from_value(meta["order"].clone()).unwrap();
            let names: Vec<String> = fixtures.iter().map(|f| f.file_name().unwrap().to_string_lossy().into()).collect();
            // The reference batched its members in its own order; compare each member's slice.
            let rows: Vec<usize> = serde_json::from_value(meta["rows"].clone()).unwrap();
            let reference = read_f32(&golden.join("batch/stage_out.f32"));
            let mut ours_start = std::collections::BTreeMap::new();
            let mut at = 0;
            for (name, grid) in names.iter().zip(&batch_grids) {
                let n = cuteafd_ffi::vision::output_rows(*grid, true) * 3072;
                ours_start.insert(name.clone(), (at, n)); at += n;
            }
            let (mut cursor, mut worst_batch) = (0usize, 1f64);
            for (name, r) in order.iter().zip(&rows) {
                let n = r * 3072;
                let (start, len) = ours_start[name];
                assert_eq!(len, n, "{name}: batch rows differ from the reference");
                let ours: Vec<f32> = batched[start..start + len].iter().map(|v| bf16(*v)).collect();
                worst_batch = worst_batch.min(compare(&ours, &reference[cursor..cursor + n]).0);
                cursor += n;
            }
            report.insert("batch_vs_reference_worst_cosine".into(), worst_batch.into());
            worst_cos = worst_cos.min(worst_batch);
        }
        report.insert("worst_cosine".into(), worst_cos.into());
        report.insert("worst_output_rel_l2".into(), worst_out.into());
        let text = serde_json::to_string_pretty(&report).unwrap();
        println!("{text}");
        if let Some(path) = std::env::var_os("HY4_VISION_REPORT") { std::fs::write(path, &text).unwrap(); }
        assert!(identical, "batched Hy4 tower output differs from single-image encodes");
        assert!(worst_cos > 0.999, "Hy4 tower stage cosine {worst_cos} below 0.999");
    }

    #[test]
    fn hy4_full_40_layer_headers_build_a_valid_generic_tower() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = serde_json::json!({
            "model_type": "hy_v4_vl",
            "image_token_id": 120120,
            "image_start_token_id": 120118,
            "image_end_token_id": 120119,
            "text_config": {
                "hidden_size": 6144,
                "qk_rope_head_dim": 64,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [12, 10, 10],
                    "rope_theta": 10000000,
                    "rope_type": "default"
                }
            },
            "vision_config": {
                "hidden_size": 1536,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "intermediate_size": 6144,
                "out_hidden_size": 6144,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "spatial_patch_size": 1,
                "num_channels": 3,
                "max_image_size": 2048,
                "cat_extra_token": 0,
                "learnable_mlp_pooling_size": 0,
                "hidden_act": "gelu",
                "interpolate_mode": "bilinear",
                "projector_type": "rmsnorm_mlp_adapter",
                "rms_norm_eps": 1e-5,
                "perceive_post_norm": false,
                "vision_full_attention": false
            }
        });
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
        let contract = cuteafd_loader::families::hy4::vision::vision_contract(&cfg).unwrap();
        let mut header = serde_json::Map::new();
        let mut end = 0u64;
        for (name, shape) in contract.tensor_shapes() {
            let start = end;
            end += shape.iter().product::<usize>() as u64 * 2;
            header.insert(name, serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[start,end]}));
        }
        let bytes = serde_json::to_vec(&header).unwrap();
        let mut f = File::create(dir.path().join("vit.safetensors")).unwrap();
        f.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&bytes).unwrap();
        f.set_len(8 + bytes.len() as u64 + end).unwrap();
        let mut map = serde_json::Map::new();
        for name in header.keys() { map.insert(name.clone(), "vit.safetensors".into()); }
        std::fs::write(dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": map})).unwrap()).unwrap();

        let spec = TowerSpec::from_snapshot(dir.path(), MAX_MERGED_ROWS).unwrap();
        assert_eq!(spec.image_family(), cuteafd_loader::media::ImageFamily::Hy4);
        assert_eq!(spec.reads.len(), 649);
        assert_eq!(
            (spec.native.hidden, spec.native.depth, spec.native.head_dim, spec.native.output_width),
            (1536, 40, 96, 6144)
        );
        assert_eq!(spec.row_markers(), 1);
    }

    #[test]
    #[ignore = "requires VISION_SNAPSHOT, VISION_LIBRARY, HY4_VISION_GOLDEN and a SM120/SM121 GPU"]
    fn hy4_full_tower_matches_the_serving_oracle_per_stage() {
        let snapshot = std::path::PathBuf::from(std::env::var("VISION_SNAPSHOT").unwrap());
        let library = std::path::PathBuf::from(std::env::var("VISION_LIBRARY").unwrap());
        let golden = std::path::PathBuf::from(std::env::var("HY4_VISION_GOLDEN").unwrap()).join("vision");
        let processor = cuteafd_loader::media::ProcessorConfig::from_snapshot(&snapshot,
            cuteafd_loader::media::ImageFamily::Hy4).unwrap();
        let lut = normalization_lut(&processor);
        let spec = TowerSpec::hy4(&snapshot, MAX_MERGED_ROWS).unwrap();
        let depth = spec.native.depth as i32;
        let out_width = spec.native.output_width as usize;
        let required = NativeVision::required(&library, &spec.native).unwrap();
        let mut runtime = VitRuntime::load(spec, &library, 0, required.total_bytes()).unwrap();
        // SAFETY: the same trusted native library the tower owner loaded.
        let native = unsafe { cuteafd_ffi::NativeLibrary::load(&library) }.unwrap();
        let meta_of = |p: &Path| -> Option<serde_json::Value> { serde_json::from_slice(&std::fs::read(p.join("meta.json")).ok()?).ok() };
        let mut fixtures: Vec<_> = std::fs::read_dir(&golden).unwrap().map(|e| e.unwrap().path())
            .filter(|p| meta_of(p).is_some_and(|m| m.get("grid").is_some())).collect();
        fixtures.sort();
        assert!(!fixtures.is_empty(), "no Hy4 vision fixtures under {}", golden.display());
        let mut report = serde_json::Map::new();
        let mut batch_rgb = Vec::new();
        let mut batch_grids = Vec::new();
        let mut singles = Vec::new();
        let (mut worst_free_cos, mut worst_free_out_rel) = (1f64, 0f64);
        let (mut worst_teacher_cos, mut worst_teacher_rel) = (1f64, 0f64);
        let (mut worst_engine_vs_f32_out, mut worst_serving_vs_f32_out) = (1f64, 1f64);
        let read_bf16 = |path: &Path| -> Vec<u16> {
            std::fs::read(path)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect()
        };
        for fixture in &fixtures {
            let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture.join("meta.json")).unwrap()).unwrap();
            let grid: [usize; 2] = serde_json::from_value(meta["grid"].clone()).unwrap();
            let rgb = std::fs::read(fixture.join("rgb.u8")).unwrap();
            let rows = cuteafd_ffi::vision::output_rows(grid, true);
            let mut output = vec![0u16; rows * out_width];
            let mut stages: Vec<(String, Vec<f32>)> = Vec::new();
            runtime.native.encode_batch_observed(&rgb, &[grid], &lut, &mut output, &mut |stage, data, rows, width| {
                let extent = rows * width * 2;
                let mut bytes = vec![0u8; extent];
                // encode_batch_observed drains the vision stream before invoking the observer callback.
                native.copy_d2h_quiescent(&mut bytes, cuteafd_ffi::CuteafdDeviceBuffer { ptr: data as *mut _, bytes: extent,
                    device_id: 0, flags: 0 }).map_err(|e| cuteafd_ffi::vision::VisionError::Runtime(e.to_string()))?;
                let name = match stage { 0 => "0".to_string(), -1 => "pos".into(), s if s == depth + 1 => format!("{}", depth + 1),
                    s if s == depth + 2 => "out".into(), s => s.to_string() };
                stages.push((name, bytes.chunks_exact(2).map(|b| bf16(u16::from_le_bytes([b[0], b[1]]))).collect()));
                Ok(())
            }).unwrap();
            let mut entry = serde_json::Map::new();
            let mut free_map = serde_json::Map::new();
            for (name, actual) in &stages {
                let path = fixture.join(format!("stage_{name}.f32"));
                if !path.is_file() { continue; }
                let (cos, max_abs, rel) = compare(actual, &read_f32(&path));
                if name == "out" {
                    worst_free_out_rel = worst_free_out_rel.max(rel);
                    let f32_path = fixture.join("stage_out_f32.f32");
                    if f32_path.is_file() {
                        let (cos_f32, max_f32, rel_f32) = compare(actual, &read_f32(&f32_path));
                        worst_engine_vs_f32_out = worst_engine_vs_f32_out.min(cos_f32);
                        entry.insert("engine_vs_f32_out".into(), serde_json::json!({
                            "cosine": cos_f32, "max_abs": max_f32, "rel_l2": rel_f32
                        }));
                    }
                }
                worst_free_cos = worst_free_cos.min(cos);
                free_map.insert(name.clone(), serde_json::json!({"cosine": cos, "max_abs": max_abs, "rel_l2": rel}));
            }
            if let Some(sv_f32) = meta.get("serving_vs_f32") {
                if let Some(c) = sv_f32.get("out").and_then(|v| v.get("cosine")).and_then(|v| v.as_f64()) {
                    worst_serving_vs_f32_out = worst_serving_vs_f32_out.min(c);
                }
                entry.insert("serving_vs_f32_out".into(), sv_f32["out"].clone());
            }
            entry.insert("free_running".into(), free_map.into());

            // Teacher-forced per-layer pass: feeds the exact serving BF16 boundary
            // tensor before each of the 40 transformer blocks to isolate per-layer
            // kernel accuracy from 40-layer BF16 residual-cancellation drift.
            let teacher_inputs: Vec<Vec<u16>> = (0..depth)
                .map(|layer| read_bf16(&fixture.join(format!("teacher_{layer}.bf16"))))
                .collect();
            let mut teacher_output = vec![0u16; rows * out_width];
            let mut teacher_stages: Vec<(String, Vec<f32>)> = Vec::new();
            runtime.native.encode_batch_teacher_observed(
                &rgb,
                &[grid],
                &lut,
                &mut teacher_output,
                &mut |next_layer, rows, width| {
                    let buf = teacher_inputs.get(usize::try_from(next_layer).ok()?)?;
                    (buf.len() == rows * width).then_some(buf.as_ptr())
                },
                &mut |stage, data, rows, width| {
                    let extent = rows * width * 2;
                    let mut bytes = vec![0u8; extent];
                    // encode_batch_teacher_observed delegates to encode_batch_observed, which drains before each callback.
                    native.copy_d2h_quiescent(&mut bytes, cuteafd_ffi::CuteafdDeviceBuffer { ptr: data as *mut _, bytes: extent,
                        device_id: 0, flags: 0 }).map_err(|e| cuteafd_ffi::vision::VisionError::Runtime(e.to_string()))?;
                    let name = match stage { 0 => "0".to_string(), -1 => "pos".into(), s if s == depth + 1 => format!("{}", depth + 1),
                        s if s == depth + 2 => "out".into(), s => s.to_string() };
                    teacher_stages.push((name, bytes.chunks_exact(2).map(|b| bf16(u16::from_le_bytes([b[0], b[1]]))).collect()));
                    Ok(())
                },
            ).unwrap();
            let mut teacher_map = serde_json::Map::new();
            let mut fixture_worst_teacher = 1f64;
            for (name, actual) in &teacher_stages {
                let path = fixture.join(format!("stage_{name}.f32"));
                if !path.is_file() { continue; }
                let (cos, max_abs, rel) = compare(actual, &read_f32(&path));
                fixture_worst_teacher = fixture_worst_teacher.min(cos);
                worst_teacher_cos = worst_teacher_cos.min(cos);
                worst_teacher_rel = worst_teacher_rel.max(rel);
                teacher_map.insert(name.clone(), serde_json::json!({"cosine": cos, "max_abs": max_abs, "rel_l2": rel}));
            }
            entry.insert("teacher_forced_worst_cosine".into(), fixture_worst_teacher.into());
            entry.insert("teacher_forced".into(), teacher_map.into());

            report.insert(fixture.file_name().unwrap().to_string_lossy().into(), entry.into());
            batch_rgb.extend_from_slice(&rgb);
            batch_grids.push(grid);
            singles.extend(output);
        }
        let mut batched = vec![0u16; singles.len()];
        runtime.native.encode_batch_into(&batch_rgb, &batch_grids, &lut, &mut batched).unwrap();
        let identical = batched == singles;
        report.insert("batch_equals_singles".into(), identical.into());
        if let Some(meta) = meta_of(&golden.join("batch")) {
            let order: Vec<String> = serde_json::from_value(meta["order"].clone()).unwrap();
            let names: Vec<String> = fixtures.iter().map(|f| f.file_name().unwrap().to_string_lossy().into()).collect();
            let rows: Vec<usize> = serde_json::from_value(meta["rows"].clone()).unwrap();
            let reference = read_f32(&golden.join("batch/stage_out.f32"));
            let mut ours_start = std::collections::BTreeMap::new();
            let mut at = 0;
            for (name, grid) in names.iter().zip(&batch_grids) {
                let n = cuteafd_ffi::vision::output_rows(*grid, true) * out_width;
                ours_start.insert(name.clone(), (at, n)); at += n;
            }
            let (mut cursor, mut worst_batch) = (0usize, 1f64);
            for (name, r) in order.iter().zip(&rows) {
                let n = r * out_width;
                let (start, len) = ours_start[name];
                assert_eq!(len, n, "{name}: batch rows differ from the reference");
                let ours: Vec<f32> = batched[start..start + len].iter().map(|v| bf16(*v)).collect();
                worst_batch = worst_batch.min(compare(&ours, &reference[cursor..cursor + n]).0);
                cursor += n;
            }
            report.insert("batch_vs_reference_worst_cosine".into(), worst_batch.into());
            worst_free_cos = worst_free_cos.min(worst_batch);
        }
        report.insert("teacher_forced_worst_cosine".into(), worst_teacher_cos.into());
        report.insert("teacher_forced_worst_rel_l2".into(), worst_teacher_rel.into());
        report.insert("free_running_worst_cosine".into(), worst_free_cos.into());
        report.insert("free_running_worst_output_rel_l2".into(), worst_free_out_rel.into());
        report.insert("engine_vs_f32_worst_output_cosine".into(), worst_engine_vs_f32_out.into());
        report.insert("serving_vs_f32_worst_output_cosine".into(), worst_serving_vs_f32_out.into());
        let text = serde_json::to_string_pretty(&report).unwrap();
        println!("{text}");
        if let Some(path) = std::env::var_os("HY4_VISION_REPORT") { std::fs::write(path, &text).unwrap(); }
        assert!(identical, "batched Hy4 tower output differs from single-image encodes");
        assert!(
            worst_teacher_cos > 0.999,
            "Hy4 40-layer teacher-forced per-layer cosine {worst_teacher_cos} below 0.999"
        );
        assert!(
            worst_engine_vs_f32_out >= worst_serving_vs_f32_out - 0.015,
            "Hy4 40-layer free-running engine_vs_f32 cosine {worst_engine_vs_f32_out} lags serving_vs_f32 {worst_serving_vs_f32_out}"
        );
    }
}
