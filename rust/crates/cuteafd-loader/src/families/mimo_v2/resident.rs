//! Header-only admission for the representations MimoLoader actually keeps.
//! Routed experts, CUDA modules/graphs, runtime workspaces and draft rings
//! remain separate reservations. No tensor data or CUDA allocation is read.
use super::projection::{MimoProjectionLayout, MimoProjectionRepresentation};
use super::{FusedQkvLayout, MimoAttention, MimoV2Config};
use crate::plan::checkpoint::{Checkpoint, CheckpointTensor};
use crate::serving_capacity::CacheGeometryError;
use crate::formats::native_dense::tensor;
use cuteafd_core::{serving_capacity::MemoryReservation, DType};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct MimoResidentOptions {
    pub layers: usize,
    pub coordinator_ranks: usize,
    pub checkpoint_tp: usize,
    pub native_mtp_layers: usize,
    pub gpu_embedding: bool,
    pub head_format: MimoProjectionRepresentation,
    /// Exact selected format of each target/MTP output tensor. Loading and
    /// admission consume the same resolved choices, including mixed sources.
    pub output_formats: BTreeMap<String, MimoProjectionRepresentation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimoResidentLayout {
    pub ranks: Vec<Vec<MemoryReservation>>,
    /// Maximum extra source storage on each physical rank while target weights
    /// load, before target KV exists. BF16 TP2 slices a whole output matrix on
    /// the lead; selected FP8 packing stages at most1024 source rows per rank.
    /// A checkpoint-native FP8 upload needs no device source staging.
    pub loading_temporary_rank_bytes: Vec<u64>,
}

fn mul(label: &'static str, dims: &[u64]) -> Result<u64, CacheGeometryError> {
    dims.iter()
        .try_fold(1u64, |value, &dim| value.checked_mul(dim))
        .ok_or(CacheGeometryError::Overflow(label))
}


fn push(costs: &mut Vec<MemoryReservation>, name: impl Into<String>, bytes: u64) {
    costs.push(MemoryReservation {
        name: name.into(),
        bytes: bytes.max(256),
    });
}

/// The accepted source grids match MimoLoader::scale_rows: 128-row blocks,
/// or (for separate Q/K/V only) restarted 128+64 blocks per 192-row head.
fn fp8_scales<'a>(
    checkpoint: &'a Checkpoint,
    name: &str,
    rows: usize,
    cols: usize,
    per_head: bool,
) -> Result<&'a CheckpointTensor, CacheGeometryError> {
    let scale_name = format!("{name}_scale_inv");
    let at = checkpoint
        .tensors
        .binary_search_by(|t| t.meta.name.cmp(&scale_name))
        .map_err(|_| CacheGeometryError::ResidentTensor {
            name: scale_name.clone(),
            what: "missing from checkpoint headers".into(),
        })?;
    let t = &checkpoint.tensors[at];
    let uniform = [rows.div_ceil(128), cols.div_ceil(128)];
    let heads = [rows / 192 * 2, cols.div_ceil(128)];
    if t.meta.dtype != DType::F32
        || (t.meta.shape != uniform && !(per_head && rows % 192 == 0 && t.meta.shape == heads))
    {
        return Err(CacheGeometryError::ResidentTensor {
            name: scale_name,
            what: format!(
                "expected FP32 {uniform:?}{}; found {:?} {:?}",
                if per_head && rows % 192 == 0 {
                    format!(" or {heads:?}")
                } else {
                    String::new()
                },
                t.meta.dtype,
                t.meta.shape
            ),
        });
    }
    tensor(checkpoint, &scale_name, &t.meta.shape, &[DType::F32])
}

fn conversion_temporary(
    checkpoint: &Checkpoint,
    t: &CheckpointTensor,
) -> Result<u64, CacheGeometryError> {
    if t.meta.dtype != DType::F8E4M3 {
        return Ok(0);
    }
    let scales = fp8_scales(
        checkpoint,
        &t.meta.name,
        t.meta.shape[0],
        t.meta.shape[1],
        true,
    )?;
    t.meta
        .byte_length
        .max(256)
        .checked_add(scales.meta.byte_length.max(256))
        .ok_or(CacheGeometryError::Overflow("MiMo FP8 conversion staging"))
}

fn fp8_copy(
    costs: &mut Vec<MemoryReservation>,
    name: &str,
    rows: u64,
    cols: u64,
    two_scale_layouts: bool,
) -> Result<(), CacheGeometryError> {
    if cols == 0 || cols % 128 != 0 {
        return Err(CacheGeometryError::ResidentTensor {
            name: name.into(),
            what: "FP8 copy width must fill 128-K blocks".into(),
        });
    }
    push(
        costs,
        format!("{name}.fp8"),
        mul("MiMo FP8 values", &[rows, cols])?,
    );
    let scales = mul("MiMo expanded FP8 scales", &[rows, cols / 128, 4])?;
    push(costs, format!("{name}.row_scale"), scales);
    if two_scale_layouts {
        push(costs, format!("{name}.k_scale"), scales);
    }
    Ok(())
}

fn projection(
    checkpoint: &Checkpoint,
    source: &CheckpointTensor,
    costs: &mut Vec<MemoryReservation>,
    name: &str,
    rows: u64,
    cols: u64,
    fp8: bool,
    scale_banks: u64,
) -> Result<u64, CacheGeometryError> {
    let representation = if fp8 {
        MimoProjectionRepresentation::Fp8
    } else {
        MimoProjectionRepresentation::Bf16
    };
    let layout =
        MimoProjectionLayout::new(rows, cols, representation, scale_banks).map_err(|error| {
            CacheGeometryError::ResidentTensor {
                name: source.meta.name.clone(),
                what: error.to_string(),
            }
        })?;
    push(
        costs,
        format!("{name}.{}", if fp8 { "fp8" } else { "bf16" }),
        layout.values,
    );
    if fp8 {
        // The allocator keeps row-major and (for O) K-major scale orders.
        let bank_bytes = layout.scales / scale_banks;
        push(costs, format!("{name}.row_scale"), bank_bytes);
        if scale_banks == 2 {
            push(costs, format!("{name}.k_scale"), bank_bytes);
        }
        if source.meta.dtype == DType::F8E4M3 {
            fp8_scales(
                checkpoint,
                &source.meta.name,
                source.meta.shape[0],
                source.meta.shape[1],
                true,
            )?;
            Ok(0)
        } else {
            Ok(layout.max_load_staging)
        }
    } else {
        conversion_temporary(checkpoint, source)
    }
}

fn small(
    checkpoint: &Checkpoint,
    costs: &mut Vec<MemoryReservation>,
    name: &str,
    shape: &[usize],
    dtype: DType,
) -> Result<(), CacheGeometryError> {
    let t = tensor(checkpoint, name, shape, &[dtype])?;
    push(costs, name, t.meta.byte_length);
    Ok(())
}

fn block(
    checkpoint: &Checkpoint,
    cfg: &MimoV2Config,
    prefix: &str,
    attention: MimoAttention,
    dense: bool,
    post: &str,
    ranks: usize,
    checkpoint_tp: usize,
    output_formats: &BTreeMap<String, MimoProjectionRepresentation>,
    costs: &mut [Vec<MemoryReservation>],
) -> Result<Vec<u64>, CacheGeometryError> {
    let share = cfg
        .head_split(ranks)
        .map_err(|_| CacheGeometryError::Unsupported {
            family: "mimo_v2",
            what: "resident attention/dense heads do not partition",
        })?;
    let fused = format!("{prefix}.self_attn.qkv_proj.weight");
    let has_fused = checkpoint
        .tensors
        .binary_search_by(|t| t.meta.name.cmp(&fused))
        .is_ok();
    let qkv_rows = if has_fused {
        let full = FusedQkvLayout::new(cfg, attention, checkpoint_tp).map_err(|error| {
            CacheGeometryError::ResidentTensor {
                name: fused.clone(),
                what: error.to_string(),
            }
        })?;
        tensor(
            checkpoint,
            &fused,
            &[full.rows(), cfg.hidden],
            &[DType::F8E4M3],
        )?;
        tensor(
            checkpoint,
            &format!("{fused}_scale_inv"),
            &[full.scale_rows(), cfg.hidden.div_ceil(128)],
            &[DType::F32],
        )?;
        if checkpoint_tp % ranks != 0 {
            return Err(CacheGeometryError::ResidentTensor {
                name: fused,
                what: "checkpoint TP does not divide coordinator ranks".into(),
            });
        }
        let layout =
            FusedQkvLayout::new(&share, attention, checkpoint_tp / ranks).map_err(|error| {
                CacheGeometryError::ResidentTensor {
                    name: fused.clone(),
                    what: error.to_string(),
                }
            })?;
        if layout.rows() * ranks != full.rows() || layout.scale_rows() * ranks != full.scale_rows() {
            return Err(CacheGeometryError::ResidentTensor {
                name: fused, what: "checkpoint shards do not divide coordinator ranks".into(),
            });
        }
        layout.program_segments(&share).map_err(|error| CacheGeometryError::ResidentTensor {
            name: fused, what: error.to_string(),
        })?.0 as u64
    } else {
        if cfg.qkv_key_stride() != cfg.head_dim {
            return Err(CacheGeometryError::ResidentTensor {
                name: format!("{prefix}.self_attn.q_proj.weight"),
                what: "this program family requires padded keys; the separate Q/K/V loader does not add them (use a supported fused qkv_proj checkpoint)".into(),
            });
        }
        let rows = [
            cfg.heads * cfg.head_dim,
            cfg.kv_heads(attention) * cfg.head_dim,
            cfg.kv_heads(attention) * cfg.v_head_dim,
        ];
        for (part, &n) in ["q", "k", "v"].iter().zip(&rows) {
            let name = format!("{prefix}.self_attn.{part}_proj.weight");
            tensor(checkpoint, &name, &[n, cfg.hidden], &[DType::F8E4M3])?;
            fp8_scales(checkpoint, &name, n, cfg.hidden, true)?;
        }
        rows.iter()
            .try_fold(0u64, |sum, &n| sum.checked_add(n as u64))
            .ok_or(CacheGeometryError::Overflow("MiMo qkv rows"))?
            / ranks as u64
    };
    let o_cols = cfg.heads * cfg.v_head_dim;
    let o_name = format!("{prefix}.self_attn.o_proj.weight");
    let fp8_o = *output_formats
        .get(&o_name)
        .ok_or_else(|| CacheGeometryError::ResidentTensor {
            name: o_name.clone(),
            what: "selected output representation is missing from the resolved loading contract"
                .into(),
        })?
        == MimoProjectionRepresentation::Fp8;
    let o_tensor = tensor(
        checkpoint,
        &o_name,
        &[cfg.hidden, o_cols],
        if ranks > 1 && !fp8_o {
            &[DType::Bf16]
        } else {
            &[DType::Bf16, DType::F8E4M3]
        },
    )?;
    let o_bytes = mul("MiMo o_proj", &[cfg.hidden as u64, o_cols as u64, 2])?;
    let intermediate = cfg.dense_intermediate;
    if dense {
        for (projection, shape) in [
            ("gate", [intermediate, cfg.hidden]),
            ("up", [intermediate, cfg.hidden]),
            ("down", [cfg.hidden, intermediate]),
        ] {
            let name = format!("{prefix}.mlp.{projection}_proj.weight");
            tensor(checkpoint, &name, &shape, &[DType::F8E4M3])?;
            fp8_scales(checkpoint, &name, shape[0], shape[1], false)?;
        }
    }
    let mut temporary = vec![0; ranks];
    for (rank, out) in costs.iter_mut().take(ranks).enumerate() {
        small(
            checkpoint,
            out,
            &format!("{prefix}.input_layernorm.weight"),
            &[cfg.hidden],
            DType::Bf16,
        )?;
        small(
            checkpoint,
            out,
            &format!("{prefix}.{post}.weight"),
            &[cfg.hidden],
            DType::Bf16,
        )?;
        fp8_copy(
            out,
            &format!("{prefix}.qkv"),
            qkv_rows,
            cfg.hidden as u64,
            true,
        )?;
        temporary[rank] = projection(
            checkpoint,
            o_tensor,
            out,
            &format!("{prefix}.o_proj"),
            cfg.hidden as u64,
            (o_cols / ranks) as u64,
            fp8_o,
            2,
        )?;
        if !fp8_o && ranks > 1 && rank == 0 {
            temporary[rank] = temporary[rank].max(o_bytes);
        }
        if attention == MimoAttention::Sliding {
            let t = tensor(
                checkpoint,
                &format!("{prefix}.self_attn.attention_sink_bias"),
                &[cfg.heads],
                &[DType::Bf16],
            )?;
            push(
                out,
                format!("{}.rank{rank}", t.meta.name),
                t.meta.byte_length / ranks as u64,
            );
        }
        if dense {
            fp8_copy(
                out,
                &format!("{prefix}.gate_up"),
                mul(
                    "MiMo fused gate/up rows",
                    &[2, (intermediate / ranks) as u64],
                )?,
                cfg.hidden as u64,
                true,
            )?;
            fp8_copy(
                out,
                &format!("{prefix}.down"),
                cfg.hidden as u64,
                (intermediate / ranks) as u64,
                true,
            )?;
        } else {
            let name = format!("{prefix}.mlp.gate.weight");
            let router = tensor(
                checkpoint,
                &name,
                &[cfg.experts, cfg.hidden],
                &[DType::Bf16, DType::F32],
            )?;
            // A F32 router becomes BF16 high+low, retaining the same byte count.
            push(out, name, router.meta.byte_length);
            small(
                checkpoint,
                out,
                &format!("{prefix}.mlp.gate.e_score_correction_bias"),
                &[cfg.experts],
                DType::F32,
            )?;
        }
    }
    Ok(temporary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::testing::{
        mimo_flash_config, mimo_flash_tensors, mimo_pro_config, mimo_pro_tensors, write_snapshot,
    };

    fn options(ranks: usize, tp: usize) -> MimoResidentOptions {
        MimoResidentOptions {
            layers: 2,
            coordinator_ranks: ranks,
            checkpoint_tp: tp,
            native_mtp_layers: 0,
            gpu_embedding: true,
            head_format: MimoProjectionRepresentation::Bf16,
            output_formats: (0..2)
                .map(|layer| {
                    (
                        format!("model.layers.{layer}.self_attn.o_proj.weight"),
                        MimoProjectionRepresentation::Bf16,
                    )
                })
                .collect(),
        }
    }
    fn fp8_options(mut options: MimoResidentOptions) -> MimoResidentOptions {
        options.head_format = MimoProjectionRepresentation::Fp8;
        for mode in options.output_formats.values_mut() {
            *mode = MimoProjectionRepresentation::Fp8;
        }
        options
    }
    fn total(layout: &MimoResidentLayout) -> u64 {
        layout.ranks.iter().flatten().map(|r| r.bytes).sum()
    }

    #[test]
    fn flash_selected_fp8_owns_one_value_copy_and_two_output_scale_banks() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(
            dir.path(),
            &mimo_flash_config(),
            &mimo_flash_tensors(),
            Some(1),
        );
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = MimoV2Config::from_hf(&checkpoint.config).unwrap();
        let selected = fp8_options(options(1, 1));
        let copies = MimoResidentLayout::new(&checkpoint, &cfg, &selected).unwrap();
        let plain = MimoResidentLayout::new(&checkpoint, &cfg, &options(1, 1)).unwrap();
        assert_eq!(total(&plain) - total(&copies), 63_168_512);
        assert!(copies.ranks[0]
            .iter()
            .all(|cost| !cost.name.ends_with("o_proj.bf16") && cost.name != "lm_head.bf16"));
        assert!(plain.ranks[0].iter().all(
            |cost| !cost.name.starts_with("lm_head.fp8") && !cost.name.ends_with("o_proj.fp8")
        ));
        for layer in 0..2 {
            let name = format!("model.layers.{layer}.o_proj.k_scale");
            assert_eq!(
                copies.ranks[0]
                    .iter()
                    .find(|cost| cost.name == name)
                    .unwrap()
                    .bytes,
                4096 * 64 * 4
            );
        }
        for rank in &copies.ranks {
            for scale in rank.iter().filter(|r| r.name.ends_with(".k_scale")) {
                let row = rank
                    .iter()
                    .find(|r| r.name == scale.name.replace(".k_scale", ".row_scale"))
                    .unwrap();
                assert_eq!(row.bytes, scale.bytes);
            }
        }
        assert_eq!(copies.loading_temporary_rank_bytes, [1024 * 8192 * 2]);
        let host = MimoResidentLayout::new(
            &checkpoint,
            &cfg,
            &MimoResidentOptions {
                gpu_embedding: false,
                ..selected
            },
        )
        .unwrap();
        assert_eq!(total(&copies) - total(&host), 64 * 4096 * 2);
    }

    #[test]
    fn head_split_reserves_padded_fused_qkv_and_lead_only_weights_without_halving_norms() {
        for (config, tensors, tp, padded_rows) in [
            (mimo_flash_config(), mimo_flash_tensors(), 1, None),
            (mimo_pro_config(), mimo_pro_tensors(), 8, Some(27648)),
        ] {
            let dir = tempfile::tempdir().unwrap();
            write_snapshot(dir.path(), &config, &tensors, Some(tp));
            let checkpoint = Checkpoint::open(dir.path()).unwrap();
            let cfg = MimoV2Config::from_hf(&config).unwrap();
            let one = MimoResidentLayout::new(&checkpoint, &cfg, &options(1, tp)).unwrap();
            let split = MimoResidentLayout::new(&checkpoint, &cfg, &options(2, tp)).unwrap();
            let router: u64 = one.ranks[0].iter().filter(|r| r.name.contains(".mlp.gate.")).map(|r| r.bytes).sum();
            assert_eq!(total(&split) - total(&one), 4 * cfg.hidden as u64 * 2 + 256 + router);
            assert!(split.ranks[1].iter().all(|r| !r.name.starts_with("lm_head")
                && !r.name.starts_with("model.embed_tokens")));
            let replica: Vec<_> = split.ranks[1].iter().filter(|r| r.name.contains(".mlp.gate.")).collect();
            assert_eq!(replica, split.ranks[0].iter().filter(|r| r.name.contains(".mlp.gate.")).collect::<Vec<_>>());
            assert_eq!(
                split.loading_temporary_rank_bytes,
                [cfg.hidden as u64 * cfg.heads as u64 * 128 * 2, 0]
            );
            if let Some(rows) = padded_rows {
                let qkv = one.ranks[0]
                    .iter()
                    .find(|r| r.name == "model.layers.0.qkv.fp8")
                    .unwrap();
                assert_eq!(qkv.bytes, rows * cfg.hidden as u64);
            }
        }
    }

    #[test]
    fn split_fp8_loading_peak_belongs_to_both_physical_ranks() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &mimo_pro_config(), &mimo_pro_tensors(), Some(8));
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = MimoV2Config::from_hf(&checkpoint.config).unwrap();
        let selected = fp8_options(options(2, 8));
        let layout = MimoResidentLayout::new(&checkpoint, &cfg, &selected).unwrap();
        assert_eq!(layout.loading_temporary_rank_bytes, [16 << 20; 2]);
        for costs in &layout.ranks {
            assert!(costs
                .iter()
                .all(|cost| !cost.name.ends_with("o_proj.bf16") && cost.name != "lm_head.bf16"));
        }
        assert!(layout.ranks[1]
            .iter()
            .all(|cost| !cost.name.starts_with("lm_head")));
    }

    #[test]
    fn mixed_output_sources_keep_each_selected_format_and_reject_missing_choice() {
        let dir = tempfile::tempdir().unwrap();
        let mut tensors = mimo_flash_tensors();
        let o = "model.layers.0.self_attn.o_proj.weight";
        tensors.iter_mut().find(|tensor| tensor.0 == o).unwrap().1 = "F8_E4M3";
        tensors.push(crate::plan::testing::t(
            format!("{o}_scale_inv"),
            "F32",
            &[32, 64],
        ));
        write_snapshot(dir.path(), &mimo_flash_config(), &tensors, Some(1));
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = MimoV2Config::from_hf(&checkpoint.config).unwrap();
        let mut selected = options(2, 1);
        selected
            .output_formats
            .insert(o.into(), MimoProjectionRepresentation::Fp8);
        let layout = MimoResidentLayout::new(&checkpoint, &cfg, &selected).unwrap();
        for rank in &layout.ranks {
            assert!(rank
                .iter()
                .any(|cost| cost.name == "model.layers.0.o_proj.fp8"));
            assert!(!rank
                .iter()
                .any(|cost| cost.name == "model.layers.0.o_proj.bf16"));
            assert!(rank
                .iter()
                .any(|cost| cost.name == "model.layers.1.o_proj.bf16"));
            assert!(!rank
                .iter()
                .any(|cost| cost.name == "model.layers.1.o_proj.fp8"));
        }
        assert_eq!(layout.loading_temporary_rank_bytes, [4096 * 8192 * 2, 0]);
        selected.output_formats.remove(o);
        assert!(
            matches!(MimoResidentLayout::new(&checkpoint,&cfg,&selected),
            Err(CacheGeometryError::ResidentTensor{name,..}) if name==o)
        );
    }

    #[test]
    fn missing_or_unsupported_resident_tensors_fail_before_any_cuda_dependency() {
        let dir = tempfile::tempdir().unwrap();
        let config = mimo_flash_config();
        let mut tensors = mimo_flash_tensors();
        tensors.retain(|t| t.0 != "model.layers.0.input_layernorm.weight");
        write_snapshot(dir.path(), &config, &tensors, Some(1));
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = MimoV2Config::from_hf(&config).unwrap();
        assert!(
            matches!(MimoResidentLayout::new(&checkpoint,&cfg,&options(1,1)),
            Err(CacheGeometryError::ResidentTensor {name,..}) if name=="model.layers.0.input_layernorm.weight")
        );
    }

    #[test]
    fn source_fp8_dequantization_reserves_staging_and_rejects_a_bad_grid() {
        let dir = tempfile::tempdir().unwrap();
        let config = mimo_flash_config();
        let mut tensors = mimo_flash_tensors();
        let o = "model.layers.0.self_attn.o_proj.weight";
        tensors.iter_mut().find(|t| t.0 == o).unwrap().1 = "F8_E4M3";
        tensors.push(crate::plan::testing::t(
            format!("{o}_scale_inv"),
            "F32",
            &[32, 64],
        ));
        write_snapshot(dir.path(), &config, &tensors, Some(1));
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = MimoV2Config::from_hf(&config).unwrap();
        let layout = MimoResidentLayout::new(&checkpoint, &cfg, &options(1, 1)).unwrap();
        assert_eq!(
            layout.loading_temporary_rank_bytes,
            [4096 * 8192 + 32 * 64 * 4]
        );
        // The BF16 TP2 loader accepts BF16 o_proj only, before slicing.
        assert!(
            matches!(MimoResidentLayout::new(&checkpoint, &cfg, &options(2, 1)),
            Err(CacheGeometryError::ResidentTensor { name, .. }) if name == o)
        );

        let fp8 = fp8_options(options(2, 1));
        let native = MimoResidentLayout::new(&checkpoint, &cfg, &fp8).unwrap();
        // Layer0 directly uploads its checkpoint FP8; the other BF16 layer's
        // drained source stage is bounded independently on both ranks.
        assert_eq!(native.loading_temporary_rank_bytes, [1024 * 4096 * 2; 2]);
        assert!(native
            .ranks
            .iter()
            .all(|costs| costs.iter().all(|cost| !cost.name.ends_with("o_proj.bf16"))));

        let k = "model.layers.0.self_attn.k_proj.weight_scale_inv";
        tensors.iter_mut().find(|t| t.0 == k).unwrap().2 = vec![7, 32];
        write_snapshot(dir.path(), &config, &tensors, Some(1));
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        assert!(
            matches!(MimoResidentLayout::new(&checkpoint, &cfg, &options(1, 1)),
            Err(CacheGeometryError::ResidentTensor { name, .. }) if name == k)
        );
    }
}

impl MimoResidentLayout {
    pub fn new(
        checkpoint: &Checkpoint,
        cfg: &MimoV2Config,
        options: &MimoResidentOptions,
    ) -> Result<Self, CacheGeometryError> {
        if options.layers == 0
            || options.layers > cfg.layers
            || ![1, 2].contains(&options.coordinator_ranks)
            || cfg.attention.len() != cfg.layers
            || cfg.dense.len() != cfg.layers
            || cfg.head_dim != 192
            || cfg.v_head_dim != 128
            || cfg.program_family().is_err()
            || options.checkpoint_tp == 0
            || cfg.full_sinks
            || !cfg.swa_sinks
        {
            return Err(CacheGeometryError::Unsupported {
                family: "mimo_v2",
                what: "resident model geometry or selected layers",
            });
        }
        let mut costs = vec![Vec::new(); options.coordinator_ranks];
        let mut temporary = vec![0; options.coordinator_ranks];
        for layer in 0..options.layers {
            let peak = block(
                checkpoint,
                cfg,
                &format!("model.layers.{layer}"),
                cfg.attention[layer],
                cfg.dense[layer],
                "post_attention_layernorm",
                options.coordinator_ranks,
                options.checkpoint_tp,
                &options.output_formats,
                &mut costs,
            )?;
            for (reserved, peak) in temporary.iter_mut().zip(peak) {
                *reserved = (*reserved).max(peak);
            }
        }
        small(
            checkpoint,
            &mut costs[0],
            "model.norm.weight",
            &[cfg.hidden],
            DType::Bf16,
        )?;
        let head = tensor(
            checkpoint,
            "lm_head.weight",
            &[cfg.vocab_size, cfg.hidden],
            &[DType::Bf16, DType::F8E4M3],
        )?;
        temporary[0] = temporary[0].max(projection(
            checkpoint,
            head,
            &mut costs[0],
            "lm_head",
            cfg.vocab_size as u64,
            cfg.hidden as u64,
            options.head_format == MimoProjectionRepresentation::Fp8,
            1,
        )?);
        if options.gpu_embedding {
            small(
                checkpoint,
                &mut costs[0],
                "model.embed_tokens.weight",
                &[cfg.vocab_size, cfg.hidden],
                DType::Bf16,
            )?;
        }
        for stage in 0..options.native_mtp_layers {
            let prefix = format!("model.mtp.layers.{stage}");
            let peak = block(
                checkpoint,
                cfg,
                &prefix,
                MimoAttention::Sliding,
                true,
                "pre_mlp_layernorm",
                1,
                options.checkpoint_tp,
                &options.output_formats,
                &mut costs[..1],
            )?;
            temporary[0] = temporary[0].max(peak[0]);
            small(
                checkpoint,
                &mut costs[0],
                &format!("{prefix}.eh_proj.weight"),
                &[cfg.hidden, 2 * cfg.hidden],
                DType::Bf16,
            )?;
            for norm in ["enorm", "hnorm", "final_layernorm"] {
                small(
                    checkpoint,
                    &mut costs[0],
                    &format!("{prefix}.{norm}.weight"),
                    &[cfg.hidden],
                    DType::Bf16,
                )?;
            }
        }
        for rank in &costs {
            rank.iter()
                .try_fold(0u64, |sum, c| sum.checked_add(c.bytes))
                .ok_or(CacheGeometryError::Overflow(
                    "MiMo resident representation sum",
                ))?;
        }
        Ok(Self {
            ranks: costs,
            loading_temporary_rank_bytes: temporary,
        })
    }
}
