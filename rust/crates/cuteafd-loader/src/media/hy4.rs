//! Hy4-flash SGLang `HYV4VLImageProcessor` defaults from the checkpoint's own
//! `processor_config.json`: bicubic, 1/255 rescale, mean/std 0.5, patch 16,
//! merge 2, temporal 1 and the 4096..4194304-pixel smart_resize budget.
use super::{ImageFamily, MediaError, ProcessorConfig, Result};
use std::path::Path;

pub(super) fn processor_config(snapshot: &Path, family: ImageFamily) -> Result<ProcessorConfig> {
    let error = |e: String| MediaError::Config(e);
    let config = crate::read_json_file(&snapshot.join("config.json")).map_err(|e| error(e.to_string()))?;
    match family {
        ImageFamily::Hy4Flash => {
            crate::families::hy4_flash::vision::VisionContract::from_config(&config)
                .map_err(|e| error(e.to_string()))?;
        }
        ImageFamily::Hy4 => {
            crate::families::hy4::vision::vision_contract(&config)
                .map_err(|e| error(e.to_string()))?;
        }
        _ => return Err(error(format!("unexpected Hy4 image family {family:?}"))),
    }
    let document: serde_json::Value = crate::read_json_file(&snapshot.join("processor_config.json"))
        .map_err(|e| error(e.to_string()))?;
    let raw = &document["image_processor"];
    let expected = ProcessorConfig::for_family(family);
    let ok = raw["image_processor_type"] == "HYV4VLImageProcessor"
        && raw["patch_size"] == 16 && raw["merge_size"] == 2 && raw["temporal_patch_size"] == 1
        && raw["spatial_patch_size"] == 1
        && raw["image_mean"] == serde_json::json!(expected.mean) && raw["image_std"] == serde_json::json!(expected.std)
        && raw["min_pixels"].as_u64() == Some(expected.min_pixels) && raw["max_pixels"].as_u64() == Some(expected.max_pixels)
        && raw["size"]["shortest_edge"].as_u64() == Some(expected.min_pixels)
        && raw["size"]["longest_edge"].as_u64() == Some(expected.max_pixels)
        && raw["rescale_factor"].as_f64() == Some(1.0 / 255.0)
        // PIL resample 1 is LANCZOS; SGLang's processor coerces it to BICUBIC.
        && matches!(raw["resample"].as_u64(), Some(1 | 3))
        && ["do_convert_rgb", "do_normalize", "do_rescale", "do_resize"].iter().all(|k| raw[*k] == true);
    if !ok {
        return Err(error("unsupported Hy4 SGLang image processor geometry/numerics".into()));
    }
    expected.validate()?;
    Ok(expected)
}

#[cfg(test)]
mod tests {
    use super::super::{DecodedPixels, EncoderId, ImageProcessor, RgbImage};
    use super::*;
    #[test]
    fn grids_and_spans_follow_smart_resize_and_newline_rows() {
        let config = ProcessorConfig::for_family(ImageFamily::Hy4Flash);
        config.validate().unwrap();
        for (height, width, grid) in [(512, 512, [32, 32]), (480, 640, [30, 40]), (4000, 3000, [146, 110]), (10, 10, [4, 4])] {
            let data = (0..height * width * 3).map(|i| (i % 251) as u8).collect();
            let prepared = config.prepare_pixels(DecodedPixels::Rgb(RgbImage { width, height, data }), EncoderId([3; 32])).unwrap();
            assert_eq!([prepared.grid.h, prepared.grid.w], grid);
            assert_eq!(prepared.tokens, (grid[0] as usize / 2) * (grid[1] as usize / 2 + 1));
            assert_eq!(prepared.rgb8.len(), grid[0] as usize * grid[1] as usize * 256 * 3);
        }
        assert_eq!(config.token_geometry().row_markers, 1);
        // The normalization LUT is (x/255 - 0.5)/0.5 in F32.
        let lut = config.normalization_parameters();
        assert_eq!((lut[0], lut[255], lut[256 + 128]), (-1.0, 1.0, ((128.0f64 / 255.0) as f32 - 0.5) / 0.5));
    }
}
