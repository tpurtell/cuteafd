//! CPU media preparation. V4.1 deliberately keeps its original processor and identity.
pub mod audio;
pub mod audio_tower;
pub mod pcm_audio;
mod decode;
mod checkerboard;
pub use checkerboard::{BlendRounding, Checkerboard};
mod exif;
mod groups;
pub mod inkling;
pub mod kimi;
pub use groups::{ImageFragment,ImagePromptLayout,PreparedImageGroup};
mod muse;
pub mod minimax;
pub mod hy4;
pub mod muse_vision;
pub mod perception_vision;
mod resize;
mod spans;
pub mod step5;
pub mod still_tower;

/// Bundled audio execution selected by the checkpoint's declaration, not its LM family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioTowerKind { MimoRvq, MusePcm }
impl AudioTowerKind {
    pub fn from_config(config: &serde_json::Value) -> std::result::Result<Option<Self>, String> {
        let Some(audio) = config.get("audio_config").filter(|value| !value.is_null()) else { return Ok(None) };
        if !audio.is_object() { return Err("audio_config must be an object".into()); }
        match audio.get("model_type") {
            Some(kind) if kind == "muse_spark_audio" => Ok(Some(Self::MusePcm)),
            Some(kind) => Err(format!("unsupported bundled audio_config.model_type {kind}; add a tower exporter/kernel")),
            None if config["model_type"] == "mimo_v2" => Ok(Some(Self::MimoRvq)),
            None => Err(format!("unsupported bundled audio_config without model_type for root {}; add a tower exporter/kernel",
                config["model_type"])),
        }
    }
}

pub use cuteafd_core::{ImageKey, MediaSpan};
pub use decode::{decode, decode_bounded, decode_pixels_bounded, AlphaPolicy, DecodePolicy};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub use spans::{ExpandedMediaPrompt, MistralSpanExpander, SpanExpander};
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
/// Decoded pixels retain alpha until the family's preprocessing chooses when
/// to composite it. RGB stays RGB, so existing families allocate no extra plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodedPixels { Rgb(RgbImage), Rgba {width:u32,height:u32,data:Vec<u8>} }
impl DecodedPixels {
    pub fn dimensions(&self) -> [u32;2] { match self {
        Self::Rgb(image) => [image.width,image.height], Self::Rgba {width,height,..} => [*width,*height] } }
    pub fn bytes(&self) -> usize { match self {Self::Rgb(image)=>image.data.len(),Self::Rgba {data,..}=>data.len()} }
    pub fn resize(self,width:u32,height:u32,filter:Resample) -> Result<Self> {
        if width==0 || height==0 || u64::from(width)*u64::from(height)>64<<20 {
            return Err(invalid("resized dimensions must be positive and at most 64 megapixels"));
        }
        Ok(match self {
            Self::Rgb(image) => Self::Rgb(resize::resize(&image,width,height,filter)),
            Self::Rgba {width:iw,height:ih,data} => {
                if data.len()!=iw as usize*ih as usize*4 {return Err(invalid("RGBA extent differs from dimensions"))}
                Self::Rgba {width,height,data:resize::resize_rgba(&data,iw,ih,width,height,filter)}
            }
        })
    }
    /// Checkerboard is a family choice made after resize (Kimi) or before any
    /// padding (Inkling), never a global decoder behavior change.
    pub fn checkerboard(self) -> RgbImage {
        Checkerboard::new(8,204,153,true,BlendRounding::Nearest)
            .expect("fixed checkerboard policy").composite(self).expect("decoded pixel extent")
    }
    pub fn into_rgb(self, alpha: AlphaPolicy) -> RgbImage {
        match self {
            Self::Rgb(image) => image,
            Self::Rgba {width,height,data} => RgbImage {width,height,data:data.chunks_exact(4).flat_map(|pixel| {
                let opacity=u32::from(pixel[3]);
                (0..3).map(move |c| match alpha {
                    AlphaPolicy::Drop => pixel[c],
                    AlphaPolicy::White => ((u32::from(pixel[c])*opacity+255*(255-opacity)+127)/255) as u8,
                })
            }).collect()},
        }
    }
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
    pub encoder: EncoderId,
    pub processor: PreprocessId,
    /// ViT patch grid, not the merged LM grid; t=1 for still images.
    pub grid: ImageGrid,
    pub rgb8: Arc<[u8]>,
    pub pixels: cuteafd_core::media::ImagePixels,
    pub tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EncoderId(pub [u8; 32]);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    Muse,
    /// Mistral Large 4's Pixtral processor: longest edge 1540, 28-pixel merged
    /// cells, no temporal axis and no pad canvas.
    Mistral,
    Inkling2,
    Step5,
    Kimi,
    MiniMax,
    /// Hy4-flash: SGLang smart_resize (32-pixel cells, 4096..4194304 px),
    /// bicubic, mean/std 0.5, no temporal axis, a newline row per merged row.
    Hy4Flash,
    /// Hy4 (`hy_v4_vl`): identical SGLang `HYV4VLImageProcessor` contract as
    /// Hy4-flash, paired with Hy4's 40-layer 1536/6144 ViT tower.
    Hy4,
}

impl ImageFamily {
    /// Span geometry serving and probe admission share: merge and whether each
    /// merged row ends in a marker row. Mistral's merge is checkpoint-declared
    /// (`ProcessorConfig::token_geometry` keeps the loaded value).
    pub fn token_geometry(self) -> cuteafd_core::media::ImageTokenGeometry {
        cuteafd_core::media::ImageTokenGeometry {
            merge: ProcessorConfig::for_family(self).merge,
            row_markers: u32::from(matches!(
                self,
                ImageFamily::Mistral | ImageFamily::Hy4Flash | ImageFamily::Hy4
            )),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FamilyProcessor {
    #[default]
    None,
    Kimi(kimi::KimiProcessor),
}
impl FamilyProcessor {
    fn is_none(&self) -> bool { matches!(self,Self::None) }
}
/// Reference processor parameters; all identity-affecting choices live here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessorConfig {
    #[serde(default, skip_serializing_if = "Resample::is_bicubic")]
    pub resample: Resample,
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
    #[serde(default, skip_serializing_if = "FamilyProcessor::is_none")]
    pub family_processor: FamilyProcessor,
}
/// Largest Qwen merged image admitted by the calibrated BF16 tower.
pub const QWEN_MAX_IMAGE_TOKENS: usize = 1024;
/// Mistral Large 4's processor `size.longest_edge` (= params.json `max_image_size`).
pub const MISTRAL_LONGEST_EDGE: u64 = 1540;
/// Mistral's LM span at the longest edge: 55 merged rows of 55 `[IMG]` plus a
/// `[IMG_BREAK]`/`[IMG_END]` each.
pub const MISTRAL_MAX_IMAGE_TOKENS: usize = 55 * 56;
const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];
pub use resize::Resample;

impl ProcessorConfig {
    /// Native feature/span ceiling. Families without a qualified fixed ceiling
    /// keep their existing policy-derived capacity rather than inventing one.
    pub fn architectural_span_ceiling(&self) -> Option<usize> {
        match self.family {
            ImageFamily::Qwen=>Some(QWEN_MAX_IMAGE_TOKENS),
            ImageFamily::Mistral=>Some(self.max_image_tokens.min(MISTRAL_MAX_IMAGE_TOKENS)),
            ImageFamily::MiniMax=>Some(crate::families::minimax_m3::vision::MAX_IMAGE_TOKENS),
            ImageFamily::Kimi=>Some(crate::families::kimi_k3::vision::MAX_IMAGE_TOKENS as usize),
            ImageFamily::Hy4Flash => Some(crate::families::hy4_flash::vision::MAX_IMAGE_TOKENS),
            ImageFamily::Hy4 => Some(crate::families::hy4::vision::MAX_IMAGE_TOKENS),
            _=>None,
        }
    }
    pub fn default_span(&self) -> usize {
        match self.family {
            ImageFamily::Kimi=>crate::families::kimi_k3::vision::MAX_IMAGE_TOKENS as usize,
            ImageFamily::Hy4Flash=>crate::families::hy4_flash::vision::MAX_IMAGE_TOKENS,
            ImageFamily::Hy4=>crate::families::hy4::vision::MAX_IMAGE_TOKENS,
            _=>4096,
        }
    }
    pub fn resolved_span_cap(&self, policy: Option<usize>) -> Result<usize> {
        let requested=policy.unwrap_or_else(|| self.default_span());
        if requested==0 {return Err(invalid("image span policy must be positive"));}
        Ok(self.architectural_span_ceiling().map_or(requested,|ceiling| requested.min(ceiling)))
    }
    /// The loaded processor's span geometry: the checkpoint's merge with the family's row-marker rule.
    pub fn token_geometry(&self) -> cuteafd_core::media::ImageTokenGeometry {
        cuteafd_core::media::ImageTokenGeometry { merge: self.merge, ..self.family.token_geometry() }
    }

    pub fn for_family(family: ImageFamily) -> Self {
        let (patch, min_pixels, max_pixels, mean, std) = match family {
            ImageFamily::Mimo => (16, 3136, 12_845_056, CLIP_MEAN, CLIP_STD),
            ImageFamily::Qwen => (16, 65_536, 16_777_216, [0.5; 3], [0.5; 3]),
            ImageFamily::GlmFlash => (14, 16, 8000, CLIP_MEAN, CLIP_STD),
            ImageFamily::Muse => (14, 1, 4096, [0.5; 3], [0.5; 3]),
            // Longest edge in pixels (min/max are edge bounds for Mistral).
            ImageFamily::Mistral => (14, 1, MISTRAL_LONGEST_EDGE, CLIP_MEAN, CLIP_STD),
            ImageFamily::Inkling2 => (20, 1, 64 << 20, CLIP_MEAN, CLIP_STD),
            ImageFamily::Step5 => (14, 1, 3024 * 3024, CLIP_MEAN, CLIP_STD),
            ImageFamily::Kimi => (14, 1, crate::families::kimi_k3::vision::MAX_IMAGE_CANVAS_PIXELS, [0.5;3], [0.5;3]),
            ImageFamily::MiniMax => (14, 28 * 28, 672 * 672, CLIP_MEAN, CLIP_STD),
            ImageFamily::Hy4Flash | ImageFamily::Hy4 => (16, crate::families::hy4_flash::vision::MIN_PIXELS,
                crate::families::hy4_flash::vision::MAX_PIXELS, [0.5; 3], [0.5; 3]),
        };
        Self {
            resample: match family {ImageFamily::Muse=>Resample::Lanczos,ImageFamily::Step5=>Resample::Bilinear,_=>Resample::Bicubic},
            family,
            patch,
            merge: if family==ImageFamily::Step5 {4} else {2},
            temporal: if matches!(family,ImageFamily::Kimi|ImageFamily::Hy4Flash|ImageFamily::Hy4) {1} else {2},
            mean,
            std,
            min_pixels,
            max_pixels,
            max_image_tokens: match family {
                ImageFamily::Qwen => QWEN_MAX_IMAGE_TOKENS,
                ImageFamily::Mistral => MISTRAL_MAX_IMAGE_TOKENS,
                ImageFamily::Kimi => crate::families::kimi_k3::vision::MAX_IMAGE_TOKENS as usize,
                ImageFamily::MiniMax => crate::families::minimax_m3::vision::MAX_IMAGE_TOKENS,
                ImageFamily::Hy4Flash => crate::families::hy4_flash::vision::MAX_IMAGE_TOKENS,
                ImageFamily::Hy4 => crate::families::hy4::vision::MAX_IMAGE_TOKENS,
                _ => 4096,
            },
            decode: DecodePolicy::default(),
            family_processor: FamilyProcessor::None,
        }
    }
    /// HF AutoProcessor reads preprocessor_config.json. It takes precedence
    /// over MiMo config.json's conflicting embedded processor_config (D3).
    pub fn from_snapshot(snapshot: &Path, family: ImageFamily) -> Result<Self> {
        if family==ImageFamily::Kimi {
            let cfg=crate::read_json_file(&snapshot.join("config.json")).map_err(|e| MediaError::Config(e.to_string()))?;
            crate::families::kimi_k3::vision::VisionContract::from_config(&cfg).map_err(|e| MediaError::Config(e.to_string()))?;
            let mut config=Self::for_family(family);
            config.family_processor=FamilyProcessor::Kimi(kimi::KimiProcessor::from_snapshot(snapshot)?);
            config.validate()?;
            return Ok(config);
        }
        if family == ImageFamily::MiniMax {
            return minimax::processor_config(snapshot);
        }
        if matches!(family, ImageFamily::Hy4Flash | ImageFamily::Hy4) {
            return hy4::processor_config(snapshot, family);
        }
        if matches!(family,ImageFamily::Inkling2|ImageFamily::Step5) {
            let cfg:serde_json::Value=crate::read_json_file(&snapshot.join("config.json"))
                .map_err(|e| MediaError::Config(e.to_string()))?;
            let geometry=perception_vision::PerceptionGeometry::from_config(&cfg).map_err(MediaError::Config)?;
            let expected=if family==ImageFamily::Inkling2 {perception_vision::PerceptionKind::Inkling2} else {perception_vision::PerceptionKind::Step5};
            if geometry.kind!=expected {return Err(MediaError::Config("PE processor/checkpoint family mismatch".into()));}
            // Step has no processor side file: its pinned official class declares
            // CLIP + bilinear728/504. Inkling's nominal0.5 describes a later boundary.
            let path=snapshot.join("processor_config.json");
            if path.exists() && family==ImageFamily::Inkling2 {
                let processor:serde_json::Value=crate::read_json_file(&path).map_err(|e| MediaError::Config(e.to_string()))?;
                if processor["image_processor"]["image_processor_type"]!="Inkling2ImageProcessor" {
                    return Err(MediaError::Config("unsupported Inkling processor class".into()));
                }
            }
            let config=Self::for_family(family);config.validate()?;return Ok(config);
        }
        // GLM's standard HF snapshot embeds image/video processors in this file.
        // MiMo/Qwen keep their existing AutoProcessor precedence unchanged.
        let file = if matches!(family, ImageFamily::GlmFlash | ImageFamily::Muse | ImageFamily::Mistral) {
            "processor_config.json"
        } else {
            "preprocessor_config.json"
        };
        let bytes = std::fs::read(snapshot.join(file))
            .map_err(|e| MediaError::Config(e.to_string()))?;
        let document: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| MediaError::Config(e.to_string()))?;
        let value = if matches!(family, ImageFamily::GlmFlash | ImageFamily::Muse | ImageFamily::Mistral) {
            document.get("image_processor").filter(|v| v.is_object()).ok_or_else(|| {
                MediaError::Config("processor_config.json requires image_processor object".into())
            })?
        } else {
            &document
        };
        let mut config = Self::for_family(family);
        if family == ImageFamily::Mistral {
            return Self::mistral_from(&document, value, config);
        }
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
        let (min_name, max_name) = if matches!(family, ImageFamily::GlmFlash | ImageFamily::Muse) {
            ("min_image_tokens", "max_image_tokens")
        } else {
            ("min_pixels", "max_pixels")
        };
        for (name, size_name, dest) in [
            (min_name, "shortest_edge", &mut config.min_pixels),
            (max_name, "longest_edge", &mut config.max_pixels),
        ] {
            let v = value.get(name).or_else(|| {
                if matches!(family, ImageFamily::GlmFlash | ImageFamily::Muse) {
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
        if let Some(value) = value.get("resample") {
            config.resample = match value.as_u64() {
                Some(3) => Resample::Bicubic,
                Some(1) => Resample::Lanczos,
                _ => return Err(MediaError::Config("unsupported Pillow resampler".into())),
            };
        }
        if family == ImageFamily::Muse {
            config.max_image_tokens = usize::try_from(config.max_pixels)
                .map_err(|_| MediaError::Config("Muse image token budget overflow".into()))?;
            for name in ["do_convert_rgb", "do_normalize", "do_rescale", "do_resize"] {
                if value.get(name).is_some_and(|v| v.as_bool() != Some(true)) {
                    return Err(MediaError::Config(format!("Muse requires {name}=true")));
                }
            }
            if value.get("rescale_factor").is_some_and(|v| v.as_f64() != Some(1.0 / 255.0)) {
                return Err(MediaError::Config("Muse requires rescale_factor=1/255".into()));
            }
            if config.patch != 14 || config.merge != 2 || config.temporal != 2
                || config.max_image_tokens > 4096 {
                return Err(MediaError::Config("unsupported Muse processor geometry or token cap".into()));
            }
        }
        config.validate()?;
        Ok(config)
    }
    /// PixtralImageProcessor (HF): `patch_size`, `size.longest_edge`, mean/std,
    /// bicubic; the processor's `spatial_merge_size` sets the 28-pixel cell.
    fn mistral_from(document: &serde_json::Value, value: &serde_json::Value, mut config: Self) -> Result<Self> {
        let int = |v: &serde_json::Value, name: &str| v.as_u64().and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| MediaError::Config(format!("{name} must be a positive integer")));
        config.patch = int(&value["patch_size"], "patch_size")?;
        config.merge = int(&document["spatial_merge_size"], "spatial_merge_size")?;
        config.max_pixels = value["size"]["longest_edge"].as_u64()
            .ok_or_else(|| MediaError::Config("size.longest_edge must be a positive integer".into()))?;
        for (name, dest) in [("image_mean", &mut config.mean), ("image_std", &mut config.std)] {
            *dest = serde_json::from_value(value[name].clone()).map_err(|e| MediaError::Config(format!("{name}: {e}")))?;
        }
        if value["resample"].as_u64() != Some(3) || value["do_resize"] != true || value["do_normalize"] != true
            || value["do_rescale"] != true || value["rescale_factor"].as_f64() != Some(1.0 / 255.0)
            || value["do_convert_rgb"] != true {
            return Err(MediaError::Config("Pixtral processor must rescale 1/255, normalize and resize bicubic".into()));
        }
        // The token cap follows the checkpoint's own longest edge.
        let cells = config.max_pixels.div_ceil(u64::from(config.patch * config.merge)) as usize;
        config.max_image_tokens = cells * (cells + 1);
        config.validate()?;
        Ok(config)
    }
    pub fn with_detail(&self, low: bool) -> Self {
        let mut config = self.clone();
        if low && config.family == ImageFamily::Mistral {
            // Pixtral bounds the edge, not the count: the largest edge whose
            // span (cells * (cells + 1)) stays within 256 rows.
            let cell = u64::from(config.patch * config.merge);
            let cells = (1..).take_while(|c: &u64| c * (c + 1) <= 256).last().unwrap_or(1);
            config.max_pixels = config.max_pixels.min(cells * cell);
            config.max_image_tokens = (cells * (cells + 1)) as usize;
        } else if low {
            config.max_image_tokens = config.max_image_tokens.min(256);
        }
        config
    }
    pub fn validate(&self) -> Result<()> {
        match (&self.family_processor,self.family) {
            (FamilyProcessor::Kimi(processor),ImageFamily::Kimi)=>processor.validate()?,
            (FamilyProcessor::None,family) if family!=ImageFamily::Kimi=>{},
            _=>return Err(MediaError::Config("family processor variant differs from image family".into())),
        }
        if self.patch == 0
            || self.merge == 0
            || self.temporal != if matches!(self.family,ImageFamily::Kimi|ImageFamily::Hy4Flash|ImageFamily::Hy4) {1} else {2}
            || self.max_image_tokens == 0
            || self.max_image_tokens > if self.family==ImageFamily::Kimi {crate::families::kimi_k3::vision::MAX_IMAGE_TOKENS as usize} else {16_384}
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
        if matches!(self.family,ImageFamily::Inkling2|ImageFamily::Step5) {
            let expected=Self::for_family(self.family);
            if self.patch!=expected.patch || self.merge!=expected.merge || self.temporal!=expected.temporal
                || self.mean!=expected.mean || self.std!=expected.std || self.resample!=expected.resample {
                return Err(MediaError::Config("PE processor numerics differ from pinned bundled class".into()));
            }
            return Ok(());
        }
        if self.family==ImageFamily::Kimi {
            let expected=Self::for_family(ImageFamily::Kimi);
            if self.patch!=expected.patch || self.merge!=expected.merge || self.mean!=expected.mean
                || self.std!=expected.std || self.resample!=expected.resample {
                return Err(MediaError::Config("Kimi processor numerics differ from bundled class".into()));
            }
            return Ok(());
        }
        if matches!(self.family, ImageFamily::Hy4Flash | ImageFamily::Hy4) {
            let expected = Self::for_family(self.family);
            if self.patch != expected.patch || self.merge != expected.merge || self.temporal != expected.temporal
                || self.mean != expected.mean || self.std != expected.std || self.resample != expected.resample
                || self.min_pixels != expected.min_pixels || self.max_pixels != expected.max_pixels
                || self.max_image_tokens > expected.max_image_tokens {
                return Err(MediaError::Config("Hy4 processor differs from the supported SGLang defaults".into()));
            }
            return Ok(());
        }
        if self.family == ImageFamily::MiniMax {
            let expected = Self::for_family(ImageFamily::MiniMax);
            if self.patch != expected.patch || self.merge != expected.merge || self.temporal != expected.temporal
                || self.mean != expected.mean || self.std != expected.std || self.resample != expected.resample
                || self.min_pixels != expected.min_pixels || self.max_pixels != expected.max_pixels
                || self.max_image_tokens > expected.max_image_tokens {
                return Err(MediaError::Config("MiniMax processor differs from supported SGLang defaults".into()));
            }
            return Ok(());
        }
        if self.family == ImageFamily::Mistral {
            return Ok(());
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
            .min(if matches!(self.family, ImageFamily::GlmFlash | ImageFamily::Muse) {
                cap
            } else {
                cap * u64::from(self.patch * self.merge).pow(2)
            })
    }
    pub fn id(&self) -> PreprocessId {
        let mut hash = Sha256::new();
        hash.update(match self.resample {
            Resample::Bicubic => b"cuteafd-preprocess-v1:pillow-bicubic".as_slice(),
            Resample::Lanczos => b"cuteafd-preprocess-v1:pillow-lanczos".as_slice(),
            Resample::Bilinear => b"cuteafd-preprocess-v1:pillow-bilinear".as_slice(),
        });
        hash.update(serde_json::to_vec(self).expect("processor parameters serialize"));
        if let Some(source)=self.normalization_source() {
            hash.update(source.as_bytes());
            for value in self.normalization_parameters() {hash.update(value.to_le_bytes());}
        }
        PreprocessId(hash.finalize().into())
    }
    /// Owner parameters are part of PE preprocessing identity. Legacy tables
    /// retain the exact LUT layout and operation order they have always used.
    pub fn normalization_parameters(&self) -> [f32;768] {
        match self.family {
            ImageFamily::Inkling2 | ImageFamily::Step5 => {
                let mut parameters=[0.0;768];
                for channel in 0..3 {
                    parameters[channel]=self.mean[channel] as f32;
                    parameters[channel+3]=if self.family==ImageFamily::Inkling2 {
                        [0.26862954f32,0.2613026,0.2757771][channel]
                    } else {self.std[channel] as f32};
                    parameters[channel+6]=self.std[channel] as f32;
                }
                parameters
            }
            _=>std::array::from_fn(|i| {let c=i/256;let pixel=((i%256) as f64*(1.0/255.0)) as f32;
                (pixel-self.mean[c] as f32)/self.std[c] as f32})
        }
    }
    pub fn normalization_source(&self) -> Option<&'static str> {
        match self.family {
            ImageFamily::Inkling2=>Some("Inkling bundled vision.py sha256=34525c3b6afb44851de1f0341f60f6006760256c5e507b18cf5167a17497117c:processor-CLIP-bf16/encoder-CLIP-remap"),
            ImageFamily::Step5=>Some("Step bundled step3_processor.py sha256=ce0265261a6bffaf40b98911ef5b9babe30d20443af83a4c8ecf251ce6a861fa:Step3VisionProcessor/Step3VLImageProcessor:CLIP-bilinear-antialias728/504"),
            ImageFamily::MiniMax=>Some("sglang@a9bf8c76 MiniMaxM31ImageProcessor:RGB-before-Pillow-bicubic:ceil28-fit672:CLIP-f32:duplicate-still"),
            ImageFamily::Hy4Flash | ImageFamily::Hy4 => Some("sglang@ccb8d9cc HYV4VLImageProcessor:Pillow-bicubic:smart_resize32-4096-4194304:mean-std0.5-f32:raster-p16"),
            _=>None,
        }
    }
    /// Prepare encoder jobs without template tokens: families may need source
    /// dimensions and prepared grids before they can assemble their prompt.
    pub fn prepare_group(&self,pixels:DecodedPixels,encoder:EncoderId) -> Result<PreparedImageGroup> {
        self.validate()?;
        if self.family==ImageFamily::Step5 {return step5::prepare_group(self,pixels,encoder);}
        let image=Arc::new(self.prepare_pixels(pixels,encoder)?);
        Ok(PreparedImageGroup {source:image.pixels.source,images:vec![image],fragments:Vec::new()})
    }
    /// A family with dimension-dependent tokens may assign `group.fragments`
    /// itself instead, before calling the checked group expansion.
    pub fn assemble_group_prompt(&self,group:&mut PreparedImageGroup,layout:&ImagePromptLayout) -> Result<()> {
        if self.family==ImageFamily::Step5 {return step5::assemble_group_prompt(group,layout);}
        if self.family==ImageFamily::Kimi {return Err(invalid("Kimi source dimensions require family-assembled prompt fragments"));}
        let ImagePromptLayout::Single {placeholder}=layout else {return Err(invalid("single-view processor needs a single prompt layout"));};
        if group.images.len()!=1 {return Err(invalid("single-view prompt needs exactly one prepared job"));}
        group.fragments=vec![ImageFragment::Features {image:0,placeholder:*placeholder}];
        Ok(())
    }
    pub fn resize_shape(&self, height: u32, width: u32) -> Result<(u32, u32)> {
        self.validate()?;
        if height == 0 || width == 0 || u64::from(height) * u64::from(width) > 64 << 20 {
            return Err(invalid(
                "dimensions must be positive and at most 64 megapixels",
            ));
        }
        let factor = self.patch * self.merge;
        if self.family==ImageFamily::Kimi {
            let FamilyProcessor::Kimi(processor)=&self.family_processor else {unreachable!("validated Kimi variant")};
            let (_,canvas)=processor.geometry(width,height)?;
            Ok((canvas[1],canvas[0]))
        } else if self.family == ImageFamily::MiniMax {
            minimax::resize_shape(height, width)
        } else if matches!(self.family, ImageFamily::Hy4Flash | ImageFamily::Hy4) {
            crate::families::hy4_flash::vision::smart_resize(height, width).map_err(|e| invalid(e.to_string()))
        } else if self.family == ImageFamily::Muse {
            muse::resize_shape(height, width, factor, self.effective_max() as u32)
                .ok_or_else(|| invalid("Muse resize dimensions overflow"))
        } else if self.family == ImageFamily::Mistral {
            Ok(mistral_resize(height, width, factor, self.max_pixels))
        } else if self.family == ImageFamily::GlmFlash {
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
/// HF `get_resize_output_image_size` (Pixtral): scale so the longest side is
/// at most `longest` (floor, unlike mistral_common's round), then each side to
/// whole `factor` cells: `(side - 1) // factor + 1`.
fn mistral_resize(height: u32, width: u32, factor: u32, longest: u64) -> (u32, u32) {
    let (mut h, mut w) = (f64::from(height), f64::from(width));
    let ratio = (h / longest as f64).max(w / longest as f64);
    if ratio > 1.0 {
        h = (h / ratio).floor();
        w = (w / ratio).floor();
    }
    let cells = |side: f64| ((side as u64).saturating_sub(1) / u64::from(factor) + 1) as u32;
    (cells(h) * factor, cells(w) * factor)
}

/// Mistral's merged grid `(rows, columns)` and LM span length `rows * (columns + 1)`.
pub fn mistral_span(grid: ImageGrid, merge: u32) -> (usize, usize, usize) {
    cuteafd_core::media::ImageTokenGeometry { merge, row_markers: 1 }
        .extent([grid.t, grid.h, grid.w]).unwrap_or((0, 0, 0))
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
    fn prepare_pixels(&self,image:DecodedPixels,encoder:EncoderId) -> Result<PreparedImage> {
        // Existing processors retain their historical composite-before-resize
        // bytes; new families can override this hook and resize RGBA first.
        self.prepare_rgb(image.into_rgb(self.config().decode.alpha),encoder)
    }
    fn prepare(&self, encoded: &[u8], encoder: EncoderId) -> Result<PreparedImage> {
        self.prepare_pixels(decode_pixels_bounded(encoded,self.config().decode,usize::MAX)?,encoder)
    }
}
impl ImageProcessor for ProcessorConfig {
    fn config(&self) -> &ProcessorConfig {
        self
    }
    fn prepare_pixels(&self,image:DecodedPixels,encoder:EncoderId) -> Result<PreparedImage> {
        if self.family==ImageFamily::Kimi {
            self.validate()?;
            let FamilyProcessor::Kimi(processor)=&self.family_processor else {unreachable!("validated Kimi variant")};
            let mut prepared=processor.prepare(image,encoder,self.max_image_tokens)?;
            prepared.processor=self.id();
            prepared.key=PreparedImage::key_for_pixels(encoder,prepared.processor,prepared.grid,&prepared.rgb8,prepared.pixels)?;
            return Ok(prepared);
        }
        if self.family==ImageFamily::Inkling2 {
            self.validate()?;
            let mut prepared=inkling::prepare(image,encoder,self.max_image_tokens)?;
            prepared.processor=self.id();
            prepared.key=PreparedImage::key_for_pixels(encoder,prepared.processor,prepared.grid,&prepared.rgb8,prepared.pixels)?;
            return Ok(prepared);
        }
        if self.family==ImageFamily::Step5 {return Err(invalid("Step images need grouped local/global preparation"));}
        self.prepare_rgb(image.into_rgb(self.decode.alpha),encoder)
    }
    fn prepare_rgb(&self, image: RgbImage, encoder: EncoderId) -> Result<PreparedImage> {
        if matches!(self.family,ImageFamily::Inkling2|ImageFamily::Kimi) {return self.prepare_pixels(DecodedPixels::Rgb(image),encoder);}
        if self.family==ImageFamily::Step5 {return Err(invalid("Step images need grouped local/global preparation"));}
        let (height, width) = self.resize_shape(image.height, image.width)?;
        if image.data.len() != image.width as usize * image.height as usize * 3 {
            return Err(invalid("RGB byte extent differs from dimensions"));
        }
        let (_, _, tokens) = self.token_geometry().extent([1, height / self.patch, width / self.patch])
            .ok_or_else(|| invalid("image grid differs from processor merge geometry"))?;
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
            let content = resize::resize(&image, cw, ch, self.resample);
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
            resize::resize(&image, width, height, self.resample)
        };
        let grid = ImageGrid {
            t: 1,
            h: height / self.patch,
            w: width / self.patch,
        };
        let processor = self.id();
        Ok(PreparedImage {
            key: PreparedImage::key_for(encoder, processor, grid, &rgb.data),
            encoder,
            processor,
            grid,
            rgb8: rgb.data.into(),
            pixels: cuteafd_core::media::ImagePixels { source:[image.width,image.height],
                ..cuteafd_core::media::ImagePixels::full(width,height) },
            tokens,
        })
    }
}

impl PreparedImage {
    /// Rebind input identity without changing the admitted encoder's cache key.
    pub fn key_for(encoder: EncoderId, processor: PreprocessId, grid: ImageGrid, rgb: &[u8]) -> ImageKey {
        let mut hash = Sha256::new();
        hash.update(b"cuteafd-image-v1");
        hash.update(encoder.0);
        hash.update(processor.0);
        for dim in [grid.t, grid.h, grid.w] { hash.update(dim.to_le_bytes()); }
        hash.update(rgb);
        ImageKey(hash.finalize().into())
    }

    /// Content-aware towers distinguish raw sentinel padding from real RGB0.
    /// Keep historical full-canvas keys byte-identical; only nontrivial content
    /// geometry adds a domain-separated identity suffix.
    pub fn key_for_pixels(encoder:EncoderId,processor:PreprocessId,grid:ImageGrid,rgb:&[u8],pixels:cuteafd_core::media::ImagePixels) -> Result<ImageKey> {
        if !pixels.valid() || u64::from(pixels.canvas[0])*u64::from(pixels.canvas[1])*3!=rgb.len() as u64 {
            return Err(invalid("image content identity differs from RGB canvas"));
        }
        let legacy=Self::key_for(encoder,processor,grid,rgb);
        if !pixels.padded() {return Ok(legacy)}
        let mut hash=Sha256::new();hash.update(b"cuteafd-image-content-v1");hash.update(legacy.0);
        for value in pixels.canvas.into_iter().chain(pixels.content) {hash.update(value.to_le_bytes());}
        Ok(ImageKey(hash.finalize().into()))
    }

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
        if config.family == ImageFamily::Muse {
            // Muse patch rows are raster ordered, with temporal then channel
            // inside each patch; spatial merging happens after the encoder.
            for row in 0..gh {
                for column in 0..gw {
                    for _ in 0..t {
                        for (channel, table) in lut.iter().enumerate() {
                            for py in 0..p {
                                for px in 0..p {
                                    let y = row * p + py;
                                    let x = column * p + px;
                                    out.push(table[self.rgb8[(y * gw * p + x) * 3 + channel] as usize]);
                                }
                            }
                        }
                    }
                }
            }
            return Ok(out);
        }
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
