//! CPU media preparation. V4.1 deliberately keeps its original processor and identity.
mod decode;
mod exif;
mod resize;
mod spans;

pub use cuteafd_core::{ImageKey, MediaSpan};
pub use decode::{decode, decode_bounded, AlphaPolicy, DecodePolicy};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub use spans::{ExpandedMediaPrompt, SpanExpander};
use std::{path::Path, sync::Arc};

#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    #[error("invalid image: {0}")]
    Invalid(String),
    #[error("image decoding failed: {0}")]
    Decode(String),
    #[error("image processor config: {0}")]
    Config(String),
}
type Result<T> = std::result::Result<T, MediaError>;
fn invalid(message: impl Into<String>) -> MediaError {
    MediaError::Invalid(message.into())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageGrid {
    pub t: u32,
    pub h: u32,
    pub w: u32,
}
#[derive(Debug, Clone)]
pub struct PreparedImage {
    pub key: ImageKey,
    /// ViT patch grid, not the merged LM grid; t=1 for still images.
    pub grid: ImageGrid,
    pub rgb8: Arc<[u8]>,
    pub tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EncoderId(pub [u8; 32]);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreprocessId(pub [u8; 32]);
fn field(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}
impl EncoderId {
    /// `headers` must contain only this tower's safetensors header entries,
    /// sorted by tensor name. No weights need to be read to establish identity.
    pub fn derive(
        family: &str,
        revision: &str,
        headers: &std::collections::BTreeMap<String, serde_json::Value>,
        numerics: u32,
        sm: u32,
    ) -> Self {
        let mut hash = Sha256::new();
        hash.update(b"cuteafd-encoder-v1");
        field(&mut hash, family.as_bytes());
        field(&mut hash, revision.as_bytes());
        let digest = Sha256::digest(serde_json::to_vec(headers).expect("JSON values serialize"));
        hash.update(digest);
        hash.update(numerics.to_le_bytes());
        hash.update(sm.to_le_bytes());
        Self(hash.finalize().into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFamily {
    Mimo,
    Qwen,
    GlmFlash,
}
/// Reference processor parameters; all identity-affecting choices live here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessorConfig {
    pub family: ImageFamily,
    pub patch: u32,
    pub merge: u32,
    pub temporal: u32,
    pub mean: [f64; 3],
    pub std: [f64; 3],
    /// Pixel budgets for MiMo/Qwen, LM token budgets for GLM.
    pub min_pixels: u64,
    pub max_pixels: u64,
    pub max_image_tokens: usize,
    pub decode: DecodePolicy,
}
const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];
impl ProcessorConfig {
    pub fn for_family(family: ImageFamily) -> Self {
        let (patch, min_pixels, max_pixels, mean, std) = match family {
            ImageFamily::Mimo => (16, 3136, 12_845_056, CLIP_MEAN, CLIP_STD),
            ImageFamily::Qwen => (16, 65_536, 16_777_216, [0.5; 3], [0.5; 3]),
            ImageFamily::GlmFlash => (14, 16, 8000, CLIP_MEAN, CLIP_STD),
        };
        Self {
            family,
            patch,
            merge: 2,
            temporal: 2,
            mean,
            std,
            min_pixels,
            max_pixels,
            max_image_tokens: 4096,
            decode: DecodePolicy::default(),
        }
    }
    /// HF AutoProcessor reads preprocessor_config.json. It takes precedence
    /// over MiMo config.json's conflicting embedded processor_config (D3).
    pub fn from_snapshot(snapshot: &Path, family: ImageFamily) -> Result<Self> {
        // GLM's standard HF snapshot embeds image/video processors in this file.
        // MiMo/Qwen keep their existing AutoProcessor precedence unchanged.
        let file = if family == ImageFamily::GlmFlash {
            "processor_config.json"
        } else {
            "preprocessor_config.json"
        };
        let bytes = std::fs::read(snapshot.join(file))
            .map_err(|e| MediaError::Config(e.to_string()))?;
        let document: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| MediaError::Config(e.to_string()))?;
        let value = if family == ImageFamily::GlmFlash {
            document.get("image_processor").filter(|v| v.is_object()).ok_or_else(|| {
                MediaError::Config("processor_config.json requires image_processor object".into())
            })?
        } else {
            &document
        };
        let mut config = Self::for_family(family);
        for (name, dest) in [
            ("patch_size", &mut config.patch),
            ("merge_size", &mut config.merge),
            ("temporal_patch_size", &mut config.temporal),
        ] {
            if let Some(v) = value.get(name) {
                *dest = v
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| {
                        MediaError::Config(format!("{name} must be a positive integer"))
                    })?;
            }
        }
        for (name, dest) in [
            ("image_mean", &mut config.mean),
            ("image_std", &mut config.std),
        ] {
            if let Some(v) = value.get(name) {
                *dest = serde_json::from_value(v.clone())
                    .map_err(|e| MediaError::Config(format!("{name}: {e}")))?;
            }
        }
        let (min_name, max_name) = if family == ImageFamily::GlmFlash {
            ("min_image_tokens", "max_image_tokens")
        } else {
            ("min_pixels", "max_pixels")
        };
        for (name, size_name, dest) in [
            (min_name, "shortest_edge", &mut config.min_pixels),
            (max_name, "longest_edge", &mut config.max_pixels),
        ] {
            let v = value.get(name).or_else(|| {
                if family == ImageFamily::GlmFlash {
                    None
                } else {
                    value.get("size").and_then(|s| s.get(size_name))
                }
            });
            if let Some(v) = v {
                *dest = v.as_u64().ok_or_else(|| {
                    MediaError::Config(format!("{name} must be a positive integer"))
                })?;
            }
        }
        if value.get("resample").is_some_and(|v| v.as_u64() != Some(3)) {
            return Err(MediaError::Config(
                "only the gated Pillow bicubic resampler is supported".into(),
            ));
        }
        config.validate()?;
        Ok(config)
    }
    pub fn with_detail(&self, low: bool) -> Self {
        let mut config = self.clone();
        if low {
            config.max_image_tokens = config.max_image_tokens.min(256);
        }
        config
    }
    pub fn validate(&self) -> Result<()> {
        if self.patch == 0
            || self.merge == 0
            || self.temporal != 2
            || self.max_image_tokens == 0
            || self.max_image_tokens > 16_384
            || self.patch > 64
            || self.merge > 8
            || self.min_pixels == 0
            || self.max_pixels < self.min_pixels
            || self.mean.iter().any(|v| !v.is_finite())
            || self.std.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(MediaError::Config(
                "invalid processor geometry, budget or normalization".into(),
            ));
        }
        let budget = self.effective_max();
        if self.min_pixels > budget {
            return Err(MediaError::Config(
                "image token cap is below the processor minimum".into(),
            ));
        }
        Ok(())
    }
    fn effective_max(&self) -> u64 {
        let cap = self.max_image_tokens as u64;
        self.max_pixels
            .min(if self.family == ImageFamily::GlmFlash {
                cap
            } else {
                cap * u64::from(self.patch * self.merge).pow(2)
            })
    }
    pub fn id(&self) -> PreprocessId {
        let mut hash = Sha256::new();
        hash.update(b"cuteafd-preprocess-v1:pillow-bicubic");
        hash.update(serde_json::to_vec(self).expect("processor parameters serialize"));
        PreprocessId(hash.finalize().into())
    }
    pub fn resize_shape(&self, height: u32, width: u32) -> Result<(u32, u32)> {
        self.validate()?;
        if height == 0 || width == 0 || u64::from(height) * u64::from(width) > 64 << 20 {
            return Err(invalid(
                "dimensions must be positive and at most 64 megapixels",
            ));
        }
        let factor = self.patch * self.merge;
        if self.family == ImageFamily::GlmFlash {
            glm_resize(height, width, factor, self.min_pixels, self.effective_max())
        } else {
            qwen_resize(height, width, factor, self.min_pixels, self.effective_max())
        }
    }
}

fn qwen_resize(height: u32, width: u32, factor: u32, min: u64, max: u64) -> Result<(u32, u32)> {
    let (h, w, f) = (f64::from(height), f64::from(width), f64::from(factor));
    if h.max(w) / h.min(w) > 200.0 {
        return Err(invalid("absolute aspect ratio must be at most 200"));
    }
    let (mut rh, mut rw) = ((h / f).round_ties_even() * f, (w / f).round_ties_even() * f);
    if rh * rw > max as f64 {
        let beta = (h * w / max as f64).sqrt();
        rh = f.max((h / beta / f).floor() * f);
        rw = f.max((w / beta / f).floor() * f);
    } else if rh * rw < min as f64 {
        let beta = (min as f64 / (h * w)).sqrt();
        rh = (h * beta / f).ceil() * f;
        rw = (w * beta / f).ceil() * f;
    }
    Ok((rh as u32, rw as u32))
}
fn glm_resize(height: u32, width: u32, factor: u32, min: u64, max: u64) -> Result<(u32, u32)> {
    let align = |v: u64| v.div_ceil(u64::from(factor)) * u64::from(factor);
    let f2 = u64::from(factor).pow(2);
    let (min, max) = (min * 2 * f2, max * 2 * f2);
    let (mut h, mut w) = (align(u64::from(height)), align(u64::from(width)));
    if 2 * h * w < min {
        let scale = (min as f64 / (2.0 * f64::from(height) * f64::from(width))).sqrt();
        h = align((f64::from(height) * scale).ceil() as u64);
        w = align((f64::from(width) * scale).ceil() as u64);
    }
    if 2 * h * w > max {
        let (mut low, mut high) = (1u64, u64::from(height));
        (h, w) = (u64::from(factor), u64::from(factor));
        while low <= high {
            let mid = (low + high) / 2;
            let candidate_h = align(mid);
            let content_w = (f64::from(width) * mid as f64 / f64::from(height)).floor() as u64;
            let candidate_w = align(content_w.max(1));
            if 2 * candidate_h * candidate_w <= max {
                (h, w) = (candidate_h, candidate_w);
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
    }
    Ok((h as u32, w as u32))
}

pub trait ImageProcessor: Send + Sync {
    fn config(&self) -> &ProcessorConfig;
    fn prepare_rgb(&self, image: RgbImage, encoder: EncoderId) -> Result<PreparedImage>;
    fn prepare(&self, encoded: &[u8], encoder: EncoderId) -> Result<PreparedImage> {
        self.prepare_rgb(decode(encoded, self.config().decode)?, encoder)
    }
}
impl ImageProcessor for ProcessorConfig {
    fn config(&self) -> &ProcessorConfig {
        self
    }
    fn prepare_rgb(&self, image: RgbImage, encoder: EncoderId) -> Result<PreparedImage> {
        let (height, width) = self.resize_shape(image.height, image.width)?;
        if image.data.len() != image.width as usize * image.height as usize * 3 {
            return Err(invalid("RGB byte extent differs from dimensions"));
        }
        let tokens = (height / (self.patch * self.merge)) as usize
            * (width / (self.patch * self.merge)) as usize;
        if tokens == 0 || tokens > self.max_image_tokens {
            return Err(invalid("official resize exceeds the image token cap"));
        }
        let rgb = if self.family == ImageFamily::GlmFlash {
            // The pinned PIL path resizes content without distortion and pads
            // the aligned canvas with black before rescaling/normalization.
            let mut scale = (f64::from(height) / f64::from(image.height))
                .min(f64::from(width) / f64::from(image.width));
            let factor = u64::from(self.patch * self.merge);
            if u64::from(image.height) * u64::from(image.width) >= factor * factor * self.min_pixels
            {
                scale = scale.min(1.0);
            }
            let ch = (f64::from(image.height) * scale)
                .floor()
                .max(1.0)
                .min(f64::from(height)) as u32;
            let cw = (f64::from(image.width) * scale)
                .floor()
                .max(1.0)
                .min(f64::from(width)) as u32;
            let content = resize::resize_bicubic(&image, cw, ch);
            let mut data = vec![0; width as usize * height as usize * 3];
            for y in 0..ch as usize {
                data[y * width as usize * 3..y * width as usize * 3 + cw as usize * 3]
                    .copy_from_slice(&content.data[y * cw as usize * 3..(y + 1) * cw as usize * 3]);
            }
            RgbImage {
                width,
                height,
                data,
            }
        } else {
            resize::resize_bicubic(&image, width, height)
        };
        let grid = ImageGrid {
            t: 1,
            h: height / self.patch,
            w: width / self.patch,
        };
        let mut hash = Sha256::new();
        hash.update(b"cuteafd-image-v1");
        hash.update(encoder.0);
        hash.update(self.id().0);
        for dim in [grid.t, grid.h, grid.w] {
            hash.update(dim.to_le_bytes());
        }
        hash.update(&rgb.data);
        Ok(PreparedImage {
            key: ImageKey(hash.finalize().into()),
            grid,
            rgb8: rgb.data.into(),
            tokens,
        })
    }
}

impl PreparedImage {
    /// Reference-compatible f32 patches. The encoder's RGB8/LUT path avoids
    /// constructing this large array; qualification uses it to check layout.
    pub fn patches(&self, config: &ProcessorConfig) -> Result<Vec<f32>> {
        config.validate()?;
        let (gh, gw, p, m, t) = (
            self.grid.h as usize,
            self.grid.w as usize,
            config.patch as usize,
            config.merge as usize,
            config.temporal as usize,
        );
        if self.grid.t != 1
            || gh == 0
            || gw == 0
            || gh % m != 0
            || gw % m != 0
            || self.rgb8.len() != gh * gw * p * p * 3
        {
            return Err(invalid("patch grid and RGB byte extent differ"));
        }
        let lut: [[f32; 256]; 3] = std::array::from_fn(|c| {
            std::array::from_fn(|v| {
                let x = (v as f64 * (1.0 / 255.0)) as f32;
                (x - config.mean[c] as f32) / config.std[c] as f32
            })
        });
        let mut out = Vec::with_capacity(gh * gw * 3 * t * p * p);
        for bh in 0..gh / m {
            for bw in 0..gw / m {
                for mh in 0..m {
                    for mw in 0..m {
                        for table in &lut {
                            for _ in 0..t {
                                for py in 0..p {
                                    for px in 0..p {
                                        let y = (bh * m + mh) * p + py;
                                        let x = (bw * m + mw) * p + px;
                                        let c = out.len() / (t * p * p) % 3;
                                        out.push(
                                            table[self.rgb8[(y * gw * p + x) * 3 + c] as usize],
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
