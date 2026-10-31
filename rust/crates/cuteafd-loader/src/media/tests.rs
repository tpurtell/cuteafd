use super::*;
fn rgb(width: u32, height: u32) -> RgbImage {
    RgbImage {
        width,
        height,
        data: (0..width as usize * height as usize * 3)
            .map(|i| (i % 251) as u8)
            .collect(),
    }
}
#[test]
fn keys_cover_pixels_geometry_policy_revision_numerics_and_arch() {
    let config = ProcessorConfig::for_family(ImageFamily::Mimo);
    let headers = std::collections::BTreeMap::from([(
        "visual.w".into(),
        serde_json::json!({"dtype":"BF16","shape":[4,8]}),
    )]);
    let encoder = EncoderId::derive("mimo", "rev", &headers, 1, 120);
    let a = config.prepare_rgb(rgb(64, 64), encoder).unwrap();
    assert_eq!(a.key, config.prepare_rgb(rgb(64, 64), encoder).unwrap().key);
    for other in [
        EncoderId::derive("mimo", "other", &headers, 1, 120),
        EncoderId::derive("mimo", "rev", &headers, 2, 120),
        EncoderId::derive("mimo", "rev", &headers, 1, 121),
    ] {
        assert_ne!(a.key, config.prepare_rgb(rgb(64, 64), other).unwrap().key);
    }
    let mut changed = rgb(64, 64);
    changed.data[0] ^= 1;
    assert_ne!(a.key, config.prepare_rgb(changed, encoder).unwrap().key);
    assert_ne!(
        a.key,
        config
            .with_detail(true)
            .prepare_rgb(rgb(64, 64), encoder)
            .unwrap()
            .key
    );
    assert_ne!(
        config.id(),
        ProcessorConfig {
            decode: DecodePolicy {
                exif_transpose: false,
                ..Default::default()
            },
            ..config.clone()
        }
        .id()
    );
    assert_ne!(
        a.key,
        config.prepare_rgb(rgb(32, 128), encoder).unwrap().key
    );
}
#[test]
fn official_budgets_caps_and_patch_layout() {
    for family in [ImageFamily::Mimo, ImageFamily::Qwen, ImageFamily::GlmFlash] {
        let config = ProcessorConfig::for_family(family);
        let image = config
            .prepare_rgb(rgb(6000, 4000), EncoderId([0; 32]))
            .unwrap();
        assert!(image.tokens <= config.max_image_tokens);
        let low = config
            .with_detail(true)
            .prepare_rgb(rgb(1000, 600), EncoderId([0; 32]))
            .unwrap();
        assert!(low.tokens <= 256);
        let patches = low.patches(&config).unwrap();
        assert_eq!(
            patches.len(),
            low.grid.h as usize * low.grid.w as usize * 3 * 2 * config.patch.pow(2) as usize
        );
        let p2 = config.patch.pow(2) as usize;
        assert_eq!(&patches[..p2], &patches[p2..2 * p2]);
    }
    assert!(ProcessorConfig::for_family(ImageFamily::Qwen)
        .resize_shape(1, 201)
        .is_err());
    assert!(ProcessorConfig::for_family(ImageFamily::Mimo)
        .resize_shape(0, 1)
        .is_err());
}
#[test]
fn qwen_snapshot_pixel_budget_cannot_raise_the_family_cap() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("preprocessor_config.json"),
        r#"{"min_pixels":65536,"max_pixels":16777216}"#).unwrap();
    let config = ProcessorConfig::from_snapshot(dir.path(), ImageFamily::Qwen).unwrap();
    assert_eq!(config.max_image_tokens, QWEN_MAX_IMAGE_TOKENS);
    assert_eq!(config.effective_max(), 1024 * 32 * 32);
    let low = config.with_detail(true);
    assert_eq!(low.max_image_tokens, 256);
    assert_eq!(low.effective_max(), 256 * 32 * 32);
    let mut old = config.clone();
    old.max_image_tokens = 4096;
    assert_ne!(old.id(), config.id());
    assert_eq!(ProcessorConfig::for_family(ImageFamily::Mimo).max_image_tokens, 4096);
    assert_eq!(ProcessorConfig::for_family(ImageFamily::GlmFlash).max_image_tokens, 4096);
}
#[test]
fn snapshot_processor_overrides_defaults_and_rejects_bad_config() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"processor_config":{"min_pixels":8192}}"#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("preprocessor_config.json"),
        r#"{"min_pixels":3136,"max_pixels":12845056}"#,
    )
    .unwrap();
    let config = ProcessorConfig::from_snapshot(dir.path(), ImageFamily::Mimo).unwrap();
    assert_eq!(config.min_pixels, 3136);
    std::fs::write(
        dir.path().join("preprocessor_config.json"),
        r#"{"patch_size":0}"#,
    )
    .unwrap();
    assert!(ProcessorConfig::from_snapshot(dir.path(), ImageFamily::Mimo).is_err());
}
#[test]
fn glm_reads_standard_nested_image_processor_without_video_parameters() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("processor_config.json"), r#"{
        "image_processor":{"patch_size":14,"temporal_patch_size":2,"merge_size":2,
            "min_image_tokens":32,"max_image_tokens":2000,"image_mean":[0.1,0.2,0.3]},
        "video_processor":{"patch_size":99,"max_image_tokens":240000}
    }"#).unwrap();
    let config = ProcessorConfig::from_snapshot(dir.path(), ImageFamily::GlmFlash).unwrap();
    assert_eq!((config.patch, config.temporal, config.merge), (14, 2, 2));
    assert_eq!((config.min_pixels, config.max_pixels), (32, 2000));
    assert_eq!(config.mean, [0.1, 0.2, 0.3]);
    assert!(ProcessorConfig::from_snapshot(dir.path(), ImageFamily::Mimo).is_err());
    for invalid in [r#"{}"#, r#"{"image_processor":[]}"#,
        r#"{"image_processor":{"patch_size":0}}"#] {
        std::fs::write(dir.path().join("processor_config.json"), invalid).unwrap();
        assert!(ProcessorConfig::from_snapshot(dir.path(), ImageFamily::GlmFlash).is_err());
    }
}
#[test]
fn span_expansion_rejects_literal_markers_and_preserves_native_ids() {
    let config = serde_json::json!({"image_token_id":12,"vision_start_token_id":11,"vision_end_token_id":13});
    let expander = SpanExpander::from_config(&config, 100).unwrap();
    let glm_config = serde_json::json!({"image_token_id":12,"image_start_token_id":11,"image_end_token_id":13});
    let glm = SpanExpander::from_config(&glm_config, 100).unwrap();
    assert_eq!((glm.placeholder, glm.start, glm.end), (12, 11, 13));
    let image = ProcessorConfig::for_family(ImageFamily::Mimo)
        .prepare_rgb(rgb(64, 64), EncoderId([0; 32]))
        .unwrap();
    let expanded = expander
        .expand(&[1, 11, 12, 13, 2], &[image.clone()], 128)
        .unwrap();
    assert_eq!(expanded.tokens, [1, 11, 12, 12, 12, 12, 13, 2]);
    assert_eq!(
        expanded.media,
        [MediaSpan {
            start: 2,
            len: 4,
            key: image.key.into()
        }]
    );
    assert!(expander.expand(&[1, 12, 2], &[image.clone()], 128).is_err());
    assert!(expander.expand(&[11, 12, 13], &[], 128).is_err());
    assert!(expander.expand(&[11, 12, 13], &[image.clone()], 5).is_err());
    assert!(expander.expand(&[101, 11, 12, 13], &[image], 128).is_err());
    assert!(SpanExpander {
        placeholder: 101,
        ..expander
    }
    .expand(&[], &[], 128)
    .is_err());
    assert!(SpanExpander {
        start: 12,
        ..expander
    }
    .expand(&[], &[], 128)
    .is_err());
}

#[test]
fn mixed_audio_image_spans_use_final_offsets_and_checkpoint_markers() {
    use super::audio::prepare_pcm;
    let config = serde_json::json!({"image_token_id":12,"vision_start_token_id":11,"vision_end_token_id":13,
        "audio_token_id":22,"audio_start_token_id":21,"audio_end_token_id":23});
    let image = ProcessorConfig::for_family(ImageFamily::Mimo)
        .prepare_rgb(rgb(64, 64), EncoderId([0; 32])).unwrap();
    let clip = prepare_pcm(vec![0.0; 4800], EncoderId([0; 32])).unwrap();
    let native = [1, 21, 22, 23, 11, 12, 13, 21, 22, 23, 2];
    let expanded = SpanExpander::expand_media(&config, 100, &native, &[image.clone()], &[clip.clone(), clip.clone()], 32).unwrap();
    assert_eq!(expanded.tokens, [1, 21, 22, 22, 23, 11, 12, 12, 12, 12, 13, 21, 22, 22, 23, 2]);
    assert_eq!(expanded.media, [MediaSpan { start: 2, len: 2, key: clip.key.into() },
        MediaSpan { start: 6, len: 4, key: image.key.into() }, MediaSpan { start: 12, len: 2, key: clip.key.into() }]);
    assert!(SpanExpander::expand_media(&config, 100, &native, &[image.clone()], &[clip.clone()], 32).is_err());
    assert!(SpanExpander::expand_media(&config, 100, &native, &[image.clone()], &[clip.clone(), clip.clone()], 15).is_err());
    assert!(SpanExpander::expand_media(&config, 100, &[22], &[], &[clip.clone()], 32).is_err());
    let mut bad = clip.clone(); bad.geometry.tokens += 1;
    assert!(SpanExpander::expand_media(&config, 100, &[21,22,23], &[], &[bad], 32).is_err());
    let old = SpanExpander::from_config(&config, 100).unwrap().expand(&[11,12,13], &[image.clone()], 32).unwrap();
    let new = SpanExpander::expand_media(&config, 100, &[11,12,13], &[image], &[], 32).unwrap();
    assert_eq!(old.tokens, new.tokens); assert_eq!(old.media, new.media);
    let text = SpanExpander::expand_media(&config, 100, &[1,2,3], &[], &[], 32).unwrap();
    assert_eq!(text.tokens, [1,2,3]); assert!(text.media.is_empty());
}

#[test]
fn decode_policy_and_header_admission() {
    let rgba = image::RgbaImage::from_raw(2, 1, vec![10, 20, 30, 0, 100, 150, 200, 128]).unwrap();
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(rgba)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    let encoded = encoded.into_inner();
    assert_eq!(
        decode(&encoded, DecodePolicy::default()).unwrap().data,
        [10, 20, 30, 100, 150, 200]
    );
    assert_eq!(
        decode(
            &encoded,
            DecodePolicy {
                alpha: AlphaPolicy::White,
                ..Default::default()
            }
        )
        .unwrap()
        .data,
        [255, 255, 255, 177, 202, 227]
    );
    assert!(decode_bounded(&encoded, DecodePolicy::default(), 5).is_err());
    assert!(decode_bounded(&encoded, DecodePolicy::default(), 6).is_ok());
}

#[test]
fn png_xmp_orientation_matches_pillow_fallback() {
    let image = image::RgbImage::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    let mut encoded = encoded.into_inner();
    let payload = b"XML:com.adobe.xmp\0\0\0\0\0<tiff:Orientation>6</tiff:Orientation>";
    let mut chunk = (payload.len() as u32).to_be_bytes().to_vec();
    chunk.extend_from_slice(b"iTXt");
    chunk.extend_from_slice(payload);
    let mut crc = !0u32;
    for byte in &chunk[4..] {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb88320 & 0u32.wrapping_sub(crc & 1));
        }
    }
    chunk.extend_from_slice(&(!crc).to_be_bytes());
    // Insert metadata before IDAT, after the fixed-length IHDR chunk.
    encoded.splice(33..33, chunk);
    let decoded = decode(&encoded, DecodePolicy::default()).unwrap();
    assert_eq!((decoded.width, decoded.height), (1, 2));
    assert_eq!(decoded.data, [1, 2, 3, 4, 5, 6]);
    let unrotated = decode(&encoded, DecodePolicy { exif_transpose: false, ..Default::default() }).unwrap();
    assert_eq!((unrotated.width, unrotated.height), (2, 1));
}


#[test]
fn legacy_processor_serialization_and_ids_stay_bicubic() {
    for family in [ImageFamily::Mimo, ImageFamily::Qwen, ImageFamily::GlmFlash] {
        let config = ProcessorConfig::for_family(family);
        let serialized = serde_json::to_vec(&config).unwrap();
        assert!(!String::from_utf8_lossy(&serialized).contains("resample"));
        let mut legacy = Sha256::new();
        legacy.update(b"cuteafd-preprocess-v1:pillow-bicubic");
        legacy.update(&serialized);
        assert_eq!(config.id(), PreprocessId(legacy.finalize().into()));
        let restored: ProcessorConfig = serde_json::from_slice(&serialized).unwrap();
        assert_eq!(restored.resample, Resample::Bicubic);
        assert_eq!(restored.id(), config.id());
        let mut lanczos = config.clone();
        lanczos.resample = Resample::Lanczos;
        assert_ne!(lanczos.id(), config.id());
        assert!(String::from_utf8_lossy(&serde_json::to_vec(&lanczos).unwrap()).contains("Lanczos"));
    }
}

fn mistral_processor() -> ProcessorConfig {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("processor_config.json"), include_str!(
        "../families/mistral_large4/official-processor-config.json")).unwrap();
    ProcessorConfig::from_snapshot(dir.path(), ImageFamily::Mistral).unwrap()
}

/// The checkpoint's own processor_config.json (@f12dabdc): PixtralImageProcessor,
/// patch 14, spatial merge 2, longest edge 1540, CLIP mean/std, bicubic.
#[test]
fn mistral_processor_reads_the_checkpoint_pixtral_config() {
    let config = mistral_processor();
    assert_eq!((config.family, config.patch, config.merge, config.max_pixels), (ImageFamily::Mistral, 14, 2, 1540));
    assert_eq!(config.max_image_tokens, 55 * 56);
    assert_eq!(config.resample, Resample::Bicubic);
    assert_eq!(config.mean, CLIP_MEAN);
    let low = config.with_detail(true);
    assert_eq!((low.max_pixels, low.max_image_tokens), (15 * 28, 15 * 16));
    assert!(low.max_image_tokens <= 256);
    let image = low.prepare_rgb(rgb(4000, 3000), EncoderId([0; 32])).unwrap();
    assert!(image.tokens <= 256);
}

/// Token layout against mistral_common 1.13.0's ImageEncoder (`([img] * w +
/// [img_break]) * h`, last -> img_end) for 264 sizes; grids follow HF's
/// Pixtral resize (floor on downscale), which differs from mistral_common's
/// round on some oversized images, so those use the HF grid and the
/// mistral_common layout rule.
#[test]
fn mistral_spans_match_mistral_common_layout_on_hf_grids() {
    use sha2::{Digest, Sha256};
    let oracle: serde_json::Value = serde_json::from_str(include_str!("mistral-span-oracle.json")).unwrap();
    let config = mistral_processor();
    let expander = MistralSpanExpander { image: 10, row_break: 12, end: 13, merge: 2, vocabulary: 100 };
    let (mut checked, mut divergent) = (0, 0);
    for case in oracle["cases"].as_array().unwrap() {
        let [h, w, mc, hf, digest] = case.as_array().unwrap().as_slice() else { panic!() };
        let (h, w) = (h.as_u64().unwrap() as u32, w.as_u64().unwrap() as u32);
        let grid = |g: &serde_json::Value| (g[0].as_u64().unwrap() as u32, g[1].as_u64().unwrap() as u32);
        let (hf, mc) = (grid(hf), grid(mc));
        // HF floors a 1-pixel side to 0 at extreme aspect; the engine clamps to one cell as mistral_common does.
        let expected = (hf.0.max(1), hf.1.max(1));
        let (rh, rw) = config.resize_shape(h, w).unwrap();
        assert_eq!((rh / 28, rw / 28), expected, "{h}x{w}");
        let image = config.prepare_rgb(rgb(w, h), EncoderId([0; 32])).unwrap();
        let expanded = expander.expand(&[1, 10, 2], std::slice::from_ref(&image), 1 << 20).unwrap();
        let span = &expanded.tokens[1..expanded.tokens.len() - 1];
        assert_eq!(expanded.media[0].len, span.len());
        if expected == mc {
            let text = format!("[{}]", span.iter().map(u32::to_string).collect::<Vec<_>>().join(", "));
            assert_eq!(hex(&Sha256::digest(text.as_bytes())[..8]), digest.as_str().unwrap(), "{h}x{w}");
            checked += 1;
        } else {
            divergent += 1;
        }
    }
    assert!(checked >= 250, "{checked} checked, {divergent} divergent");
}

fn hex(bytes: &[u8]) -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() }

#[test]
fn mistral_expansion_refuses_bare_markers_and_mismatched_spans() {
    let config = mistral_processor();
    let image = config.prepare_rgb(rgb(56, 84), EncoderId([0; 32])).unwrap();
    assert_eq!((image.grid.h, image.grid.w, image.tokens), (6, 4, 3 * 3));
    let expander = MistralSpanExpander { image: 10, row_break: 12, end: 13, merge: 2, vocabulary: 100 };
    let expanded = expander.expand(&[1, 10, 2, 10], &[image.clone(), image.clone()], 64).unwrap();
    assert_eq!(expanded.tokens, [1, 10, 10, 12, 10, 10, 12, 10, 10, 13, 2, 10, 10, 12, 10, 10, 12, 10, 10, 13]);
    assert_eq!(expanded.media.iter().map(|m| (m.start, m.len)).collect::<Vec<_>>(), [(1, 9), (11, 9)]);
    for bad in [&[1, 12, 10][..], &[13, 10]] {
        assert!(expander.expand(bad, std::slice::from_ref(&image), 64).is_err());
    }
    assert!(expander.expand(&[10], &[], 64).is_err());
    assert!(expander.expand(&[10], std::slice::from_ref(&image), 8).is_err());
    let mut wrong = image.clone();
    wrong.tokens += 1;
    assert!(expander.expand(&[10], &[wrong], 64).is_err());
}
