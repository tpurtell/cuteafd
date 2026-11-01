//! Mistral Large 4's bundled Pixtral tower: official geometry, consolidated
//! tensor inventory, resident byte count, image markers and LM span layout.
//! Shared by the planner (charge, refusal) and the serving tower spec.
use std::path::Path;

/// Native tower selector (numerics 2): Qwen = 2, GLM Flash = 3, Mistral = 4
/// (`cuteafd_ffi::vision::MISTRAL_RESERVED`).
pub const RESERVED: u32 = 4;
pub const HIDDEN: usize = 2048;
pub const DEPTH: usize = 30;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = 128;
pub const INTERMEDIATE: usize = 6144;
pub const PATCH: usize = 14;
pub const MERGE: usize = 2;
pub const MERGED: usize = HIDDEN * MERGE * MERGE;
pub const LM_HIDDEN: usize = 6144;
/// Largest merged image (`[IMG]` rows) the resident tower admits: 1540 / 28 = 55 per
/// side, so 55 x 55. The span adds one `[IMG_BREAK]`/`[IMG_END]` per merged row.
pub const MAX_MERGED_TOKENS: usize = 55 * 55;

/// The tower's merged-row capacity for an LM span cap: the most `[IMG]` rows
/// (h * w) of any grid with h, w <= 55 whose span h * (w + 1) fits the cap.
pub fn merged_capacity(span_tokens: usize) -> usize {
    (1..=55).map(|h| h * (span_tokens / h).saturating_sub(1).min(55)).max().unwrap_or(0)
}

/// The checkpoint's image markers: `vision_encoder` ids from params.json, the
/// placeholder also named by config.json's `image_token_index`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Markers {
    pub image: u32,
    pub row_break: u32,
    pub end: u32,
}

impl Markers {
    pub fn from_snapshot(snapshot: &Path) -> Result<Self, String> {
        let read = |name: &str| -> Result<serde_json::Value, String> { crate::read_json_file(&snapshot.join(name))
            .map_err(|e| format!("{name}: {e}")) };
        let (config, params) = (read("config.json")?, read("params.json")?);
        let v = &params["vision_encoder"];
        let id = |value: &serde_json::Value, name: &str| -> Result<u32, String> {
            value.as_u64().and_then(|v| u32::try_from(v).ok()).ok_or_else(|| format!("Mistral {name} id missing"))
        };
        let markers = Self { image: id(&v["image_token_id"], "image_token_id")?,
            row_break: id(&v["image_break_token_id"], "image_break_token_id")?,
            end: id(&v["image_end_token_id"], "image_end_token_id")? };
        if id(&config["image_token_index"], "image_token_index")? != markers.image
            || markers.image == markers.row_break || markers.image == markers.end || markers.row_break == markers.end {
            return Err("Mistral image marker ids disagree".into());
        }
        Ok(markers)
    }
}

/// The official tower geometry, before any tensor is read; `max_tokens` is the merged capacity.
pub fn validate(config: &serde_json::Value, max_tokens: usize) -> Result<(), String> {
    let v = &config["vision_config"];
    for (name, expected) in [("hidden_size", HIDDEN), ("num_hidden_layers", DEPTH), ("num_attention_heads", HEADS),
        ("head_dim", HEAD_DIM), ("intermediate_size", INTERMEDIATE), ("patch_size", PATCH), ("num_channels", 3)] {
        if v[name].as_u64() != Some(expected as u64) {
            return Err(format!("vision_config.{name}={}; need {expected}; add a Pixtral tower kernel", v[name]));
        }
    }
    let rope = &v["rope_parameters"];
    if config["model_type"] != "mistral_large4" || v["model_type"] != "pixtral" || v["hidden_act"] != "silu"
        || rope["rope_type"] != "axial" || rope["rope_theta"].as_f64() != Some(10000.0)
        || config["spatial_merge_size"].as_u64() != Some(MERGE as u64)
        || config["projector_hidden_act"] != "gelu" || config["multimodal_projector_bias"] != false
        || config["vision_feature_layer"].as_i64() != Some(-1)
        || config["text_config"]["hidden_size"].as_u64() != Some(LM_HIDDEN as u64)
        || config["text_config"]["rms_norm_eps"].as_f64() != Some(1e-5)
        || max_tokens == 0 || max_tokens > MAX_MERGED_TOKENS {
        return Err("Mistral Large 4 vision/LM geometry or image capacity".into());
    }
    Ok(())
}

/// Every tower tensor in consolidated naming with its BF16 shape (the
/// checkpoint's `modules_to_not_convert` keeps all of them BF16).
pub fn tower_tensors() -> Vec<(String, Vec<usize>)> {
    let mut out = vec![("vision_encoder.patch_conv.weight".into(), vec![HIDDEN, 3, PATCH, PATCH]),
        ("vision_encoder.ln_pre.weight".into(), vec![HIDDEN])];
    for i in 0..DEPTH {
        let p = format!("vision_encoder.transformer.layers.{i}");
        for (name, shape) in [("attention.wq.weight", vec![HIDDEN, HIDDEN]), ("attention.wk.weight", vec![HIDDEN, HIDDEN]),
            ("attention.wv.weight", vec![HIDDEN, HIDDEN]), ("attention.wo.weight", vec![HIDDEN, HIDDEN]),
            ("attention_norm.weight", vec![HIDDEN]), ("ffn_norm.weight", vec![HIDDEN]),
            ("feed_forward.w1.weight", vec![INTERMEDIATE, HIDDEN]), ("feed_forward.w3.weight", vec![INTERMEDIATE, HIDDEN]),
            ("feed_forward.w2.weight", vec![HIDDEN, INTERMEDIATE])] {
            out.push((format!("{p}.{name}"), shape));
        }
    }
    out.extend([("pre_mm_projector_norm.weight".into(), vec![HIDDEN]),
        ("patch_merger.merging_layer.weight".into(), vec![HIDDEN, MERGED]),
        ("vision_language_adapter.w_in.weight".into(), vec![LM_HIDDEN, HIDDEN]),
        ("vision_language_adapter.w_out.weight".into(), vec![LM_HIDDEN, LM_HIDDEN])]);
    out
}

/// Resident arena bytes: every tensor BF16 as stored (the shared RMS program
/// reads BF16 gammas), the two marker embedding rows, and the 64 FP32 axial
/// frequencies. Each extent is 256-byte aligned. Header-only; the planner charges this.
pub fn resident_weight_bytes() -> u64 {
    let align = |n: u64| (n + 255) & !255;
    let mut bytes = 0;
    for (name, shape) in tower_tensors() {
        let _ = name;
        bytes = align(bytes) + shape.iter().product::<usize>() as u64 * 2;
    }
    bytes = align(bytes) + 2 * LM_HIDDEN as u64 * 2;
    align(bytes) + (HEAD_DIM / 2) as u64 * 4
}

/// Planner charge: every tower tensor present, BF16 at its official shape, and
/// the geometry the native tower supports.
pub fn resident_weights(checkpoint: &crate::plan::Checkpoint) -> Result<u64, String> {
    validate(&checkpoint.config, MAX_MERGED_TOKENS)?;
    let tensors: std::collections::BTreeMap<_, _> = checkpoint.tensors.iter().map(|t| (t.meta.name.as_str(), &t.meta)).collect();
    for (name, shape) in tower_tensors() {
        let tensor = tensors.get(name.as_str()).ok_or_else(|| format!("missing {name}"))?;
        if tensor.dtype != cuteafd_core::DType::Bf16 || tensor.shape != shape
            || tensor.byte_length != (shape.iter().product::<usize>() * 2) as u64 {
            return Err(format!("{name} needs BF16 {shape:?}"));
        }
    }
    Ok(resident_weight_bytes())
}

/// Axial frequencies in Pixtral's interleaved order: `inv_freq[0::2]` (height)
/// then `inv_freq[1::2]` (width), 32 each, over a head of 128.
pub fn axial_inv_freq() -> [f32; HEAD_DIM / 2] {
    let base: Vec<f32> = (0..HEAD_DIM / 2).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / HEAD_DIM as f32)).collect();
    std::array::from_fn(|i| if i < HEAD_DIM / 4 { base[2 * i] } else { base[2 * (i - HEAD_DIM / 4) + 1] })
}

/// Mistral's LM span for a merged `h x w` grid (`mistral_common` ImageEncoder):
/// each row `[IMG]` x w then `[IMG_BREAK]`, the final token `[IMG_END]`.
pub fn span_tokens(markers: Markers, h: usize, w: usize) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(h * (w + 1));
    for _ in 0..h {
        tokens.extend(std::iter::repeat_n(markers.image, w));
        tokens.push(markers.row_break);
    }
    if let Some(last) = tokens.last_mut() { *last = markers.end; }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(cells: u32) -> crate::media::PreparedImage {
        crate::media::PreparedImage { key: crate::media::ImageKey([1; 32]),
            grid: crate::media::ImageGrid { t: 1, h: cells, w: cells },
            rgb8: std::sync::Arc::from(Vec::new()), tokens: (cells * (cells + 1)) as usize }
    }

    fn config() -> serde_json::Value {
        serde_json::from_str(include_str!("official-config.json")).unwrap()
    }

    #[test]
    fn merged_capacity_is_the_largest_grid_whose_span_fits() {
        assert_eq!(merged_capacity(55 * 56), MAX_MERGED_TOKENS);
        // 5 x 47 (span 240) holds more [IMG] rows than the square 15 x 15.
        assert_eq!(merged_capacity(15 * 16), 235);
        assert_eq!(merged_capacity(2), 1);
        assert_eq!(merged_capacity(1), 0);
        for cap in 1..400 {
            let best = (1..=55).flat_map(|h| (1..=55).map(move |w| (h, w)))
                .filter(|(h, w)| h * (w + 1) <= cap).map(|(h, w)| h * w).max().unwrap_or(0);
            assert_eq!(merged_capacity(cap), best, "{cap}");
        }
    }

    #[test]
    fn official_config_validates_and_unknown_geometry_refuses() {
        assert!(validate(&config(), MAX_MERGED_TOKENS).is_ok());
        for cap in [0, MAX_MERGED_TOKENS + 1] { assert!(validate(&config(), cap).is_err()); }
        for (name, value) in [("num_attention_heads", serde_json::json!(8)), ("patch_size", serde_json::json!(16)),
            ("hidden_act", serde_json::json!("gelu")), ("head_dim", serde_json::json!(64))] {
            let mut cfg = config();
            cfg["vision_config"][name] = value;
            assert!(validate(&cfg, 64).is_err(), "{name}");
        }
        let mut cfg = config();
        cfg["multimodal_projector_bias"] = serde_json::json!(true);
        assert!(validate(&cfg, 64).is_err());
    }

    /// mistral_common 1.13.0 `ImageEncoder.__call__`: ([img] * w + [img_break]) * h, last -> img_end.
    #[test]
    fn span_layout_matches_mistral_common() {
        let m = Markers { image: 10, row_break: 12, end: 13 };
        assert_eq!(span_tokens(m, 1, 1), [10, 13]);
        assert_eq!(span_tokens(m, 1, 3), [10, 10, 10, 13]);
        assert_eq!(span_tokens(m, 2, 2), [10, 10, 12, 10, 10, 13]);
        let t = span_tokens(m, 55, 55);
        assert_eq!(t.len(), 55 * 56);
        assert_eq!(t, crate::media::MistralSpanExpander { image: 10, row_break: 12, end: 13, merge: 1, vocabulary: 99 }
            .expand(&[10], &[square(55)], 1 << 20).unwrap().tokens);
        assert_eq!(t.iter().filter(|&&id| id == 10).count(), MAX_MERGED_TOKENS);
        assert_eq!(t.iter().filter(|&&id| id == 12).count(), 54);
    }

    /// HF `PixtralVisionRotaryEmbedding.compute_axial_rope_parameters` at head 128.
    #[test]
    fn axial_frequencies_follow_pixtral_interleave() {
        let f = axial_inv_freq();
        assert_eq!(f[0], 1.0);
        assert_eq!(f[1], 1.0 / 10000f32.powf(4.0 / 128.0));
        assert_eq!(f[32], 1.0 / 10000f32.powf(2.0 / 128.0));
        assert_eq!(f[63], 1.0 / 10000f32.powf(126.0 / 128.0));
    }

    #[test]
    fn planner_charges_the_complete_official_tower() {
        let dir = tempfile::tempdir().unwrap();
        crate::plan::testing::write_mistral_large4_snapshot(dir.path(), Some(&[1]), true);
        let checkpoint = crate::plan::Checkpoint::open(dir.path()).unwrap();
        let bytes = resident_weights(&checkpoint).unwrap();
        assert_eq!(bytes, resident_weight_bytes());
        // 30 x (4 x 2048^2 + 3 x 6144 x 2048) BF16 + patch, merger, adapter and two marker rows.
        assert_eq!(bytes, 3_408_462_080);
        let markers = Markers::from_snapshot(dir.path()).unwrap();
        assert_eq!(markers, Markers { image: 10, row_break: 12, end: 13 });
    }
}
