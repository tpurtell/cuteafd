use super::{invalid, MediaSpan, PreparedImage, Result};

#[derive(Debug, Clone, Copy)]
pub struct SpanExpander {
    pub placeholder: u32,
    pub start: u32,
    pub end: u32,
    pub vocabulary: u32,
}
#[derive(Debug)]
pub struct ExpandedMediaPrompt {
    /// Only these native ids reach the LM, PLE and drafters.
    pub tokens: Vec<u32>,
    pub media: Vec<MediaSpan>,
}
impl SpanExpander {
    /// Marker ids come from the checkpoint config, never hard-coded by the API.
    pub fn from_config(config: &serde_json::Value, vocabulary: u32) -> Result<Self> {
        let id = |names: &[&str]| {
            names
                .iter()
                .find_map(|n| config.get(*n))
                .and_then(|v| v.as_u64())
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| invalid(format!("missing image marker: {}", names[0])))
        };
        let expander = Self {
            placeholder: id(&["image_token_id"])?,
            start: id(&["vision_start_token_id", "image_start_token_id", "boi_token_id"])?,
            end: id(&["vision_end_token_id", "image_end_token_id", "eoi_token_id"])?,
            vocabulary,
        };
        if vocabulary >= 1 << 31
            || [expander.placeholder, expander.start, expander.end]
                .iter()
                .any(|v| *v >= vocabulary)
            || expander.placeholder == expander.start
            || expander.placeholder == expander.end
            || expander.start == expander.end
        {
            return Err(invalid("invalid image marker ids or vocabulary"));
        }
        Ok(expander)
    }
    pub fn from_audio_config(config: &serde_json::Value, vocabulary: u32) -> Result<Self> {
        let image_markers = serde_json::json!({
            "image_token_id": config.get("audio_token_id"),
            "vision_start_token_id": config.get("audio_start_token_id"),
            "vision_end_token_id": config.get("audio_end_token_id"),
        });
        Self::from_config(&image_markers, vocabulary)
    }

    /// Expand both modalities in one pass so every span uses final LM offsets.
    pub fn expand_media(
        config: &serde_json::Value,
        vocabulary: u32,
        tokens: &[u32],
        images: &[PreparedImage],
        audio: &[super::audio::PreparedAudio],
        context_limit: usize,
    ) -> Result<ExpandedMediaPrompt> {
        let image_markers = config.get("image_token_id").map(|_| Self::from_config(config, vocabulary)).transpose()?;
        let audio_markers = config.get("audio_token_id").map(|_| Self::from_audio_config(config, vocabulary)).transpose()?;
        if vocabulary == 0 || vocabulary >= 1 << 31 || tokens.iter().any(|&id| id >= vocabulary) {
            return Err(invalid("token outside checkpoint vocabulary"));
        }
        if let (Some(image), Some(audio)) = (image_markers, audio_markers) {
            if [image.placeholder, image.start, image.end].iter().any(|id|
                [audio.placeholder, audio.start, audio.end].contains(id)) {
                return Err(invalid("image and audio marker ids overlap"));
            }
        }
        let mut descriptors = Vec::new();
        if let Some(markers) = image_markers {
            descriptors.push((markers, images.iter().map(|image| (image.tokens, image.key.into())).collect::<Vec<(usize, cuteafd_core::MediaKey)>>()));
        } else if !images.is_empty() { return Err(invalid("checkpoint has no image markers")); }
        if audio.len() > super::audio::MAX_CLIPS || audio.iter().try_fold(0usize,
            |samples, clip| samples.checked_add(clip.pcm.len())).is_none_or(|samples| samples > super::audio::MAX_REQUEST_SAMPLES) {
            return Err(invalid("audio request exceeds clip count or total duration"));
        }
        if let Some(markers) = audio_markers {
            let mut rows = Vec::new();
            for clip in audio {
                let geometry = super::audio::AudioGeometry::for_policy(clip.pcm.len(), clip.geometry.policy)
                    .map_err(|e| invalid(e.to_string()))?;
                if geometry != clip.geometry { return Err(invalid("audio geometry differs from PCM")); }
                rows.push((geometry.tokens, clip.key.into()));
            }
            descriptors.push((markers, rows));
        } else if !audio.is_empty() { return Err(invalid("checkpoint has no audio markers")); }
        let mut expanded_len = tokens.len();
        for (markers, rows) in &descriptors {
            if tokens.iter().filter(|&&id| id == markers.placeholder).count() != rows.len() {
                return Err(invalid("media placeholder count differs from supplied inputs"));
            }
            for (len, _) in rows {
                if *len == 0 { return Err(invalid("media has no LM rows")); }
                expanded_len = expanded_len.checked_add(len - 1).ok_or_else(|| invalid("media prompt length overflow"))?;
            }
        }
        if expanded_len > context_limit { return Err(invalid("expanded media prompt exceeds context limit")); }
        let mut expanded = Vec::with_capacity(expanded_len);
        let mut media = Vec::with_capacity(images.len() + audio.len());
        let mut positions = vec![0; descriptors.len()];
        for (i, &token) in tokens.iter().enumerate() {
            if let Some(kind) = descriptors.iter().position(|(markers, _)| markers.placeholder == token) {
                let (markers, rows) = &descriptors[kind];
                if i == 0 || tokens.get(i - 1) != Some(&markers.start) || tokens.get(i + 1) != Some(&markers.end) {
                    return Err(invalid("media placeholder must be enclosed by its start/end tokens"));
                }
                let (len, key) = rows[positions[kind]];
                positions[kind] += 1;
                media.push(MediaSpan { start: expanded.len(), len, key });
                expanded.extend(std::iter::repeat_n(token, len));
            } else { expanded.push(token); }
        }
        Ok(ExpandedMediaPrompt { tokens: expanded, media })
    }

    pub fn expand(
        &self,
        tokens: &[u32],
        images: &[PreparedImage],
        context_limit: usize,
    ) -> Result<ExpandedMediaPrompt> {
        if self.vocabulary == 0
            || self.vocabulary >= 1 << 31
            || [self.placeholder, self.start, self.end]
                .iter()
                .any(|&v| v >= self.vocabulary)
            || self.placeholder == self.start
            || self.placeholder == self.end
            || self.start == self.end
        {
            return Err(invalid("invalid image marker ids or vocabulary"));
        }
        if tokens.iter().any(|&v| v >= self.vocabulary) {
            return Err(invalid("token outside checkpoint vocabulary"));
        }
        let count = tokens.iter().filter(|&&v| v == self.placeholder).count();
        if count != images.len() {
            return Err(invalid(
                "image placeholder count differs from supplied images",
            ));
        }
        let mut expanded_len = tokens.len();
        for image in images {
            if image.tokens == 0 {
                return Err(invalid("image has no LM rows"));
            }
            expanded_len = expanded_len
                .checked_add(image.tokens - 1)
                .ok_or_else(|| invalid("image prompt length overflow"))?;
        }
        if expanded_len > context_limit {
            return Err(invalid("expanded image prompt exceeds context limit"));
        }
        let mut expanded = Vec::with_capacity(expanded_len);
        let mut media = Vec::with_capacity(images.len());
        for (i, &token) in tokens.iter().enumerate() {
            if token != self.placeholder {
                expanded.push(token);
                continue;
            }
            if i == 0
                || tokens.get(i - 1) != Some(&self.start)
                || tokens.get(i + 1) != Some(&self.end)
            {
                return Err(invalid(
                    "image placeholder must be enclosed by image start/end tokens",
                ));
            }
            let image = &images[media.len()];
            media.push(MediaSpan {
                start: expanded.len(),
                len: image.tokens,
                key: image.key.into(),
            });
            expanded.extend(std::iter::repeat_n(self.placeholder, image.tokens));
        }
        Ok(ExpandedMediaPrompt {
            tokens: expanded,
            media,
        })
    }
}

/// Mistral's image layout (`mistral_common` ImageEncoder, HF PixtralProcessor):
/// the chat template writes one `[IMG]` per image; each becomes its merged
/// grid's rows, `[IMG]` x w then `[IMG_BREAK]`, the last token `[IMG_END]`.
/// One [`MediaSpan`] covers all h * (w + 1) positions; the tower fills the
/// marker positions with the checkpoint's own marker embedding rows.
#[derive(Debug, Clone, Copy)]
pub struct MistralSpanExpander {
    pub image: u32,
    pub row_break: u32,
    pub end: u32,
    pub merge: u32,
    pub vocabulary: u32,
}
impl MistralSpanExpander {
    pub fn expand(&self, tokens: &[u32], images: &[PreparedImage], context_limit: usize)
        -> Result<ExpandedMediaPrompt> {
        let ids = [self.image, self.row_break, self.end];
        if self.vocabulary == 0 || self.vocabulary >= 1 << 31 || self.merge == 0
            || ids.iter().any(|&v| v >= self.vocabulary)
            || self.image == self.row_break || self.image == self.end || self.row_break == self.end {
            return Err(invalid("invalid Mistral image marker ids or vocabulary"));
        }
        if tokens.iter().any(|&v| v >= self.vocabulary) {
            return Err(invalid("token outside checkpoint vocabulary"));
        }
        // The rendered prompt carries bare [IMG] placeholders only; markers are the processor's.
        if tokens.iter().any(|&v| v == self.row_break || v == self.end) {
            return Err(invalid("[IMG_BREAK]/[IMG_END] may only come from image expansion"));
        }
        if tokens.iter().filter(|&&v| v == self.image).count() != images.len() {
            return Err(invalid("image placeholder count differs from supplied images"));
        }
        let mut expanded_len = tokens.len();
        let mut grids = Vec::with_capacity(images.len());
        for image in images {
            let (h, w, len) = super::mistral_span(image.grid, self.merge);
            if image.grid.t != 1 || h == 0 || w == 0 || len != image.tokens {
                return Err(invalid("Mistral image span differs from its merged grid"));
            }
            grids.push((h, w));
            expanded_len = expanded_len.checked_add(len - 1).ok_or_else(|| invalid("image prompt length overflow"))?;
        }
        if expanded_len > context_limit {
            return Err(invalid("expanded image prompt exceeds context limit"));
        }
        let mut expanded = Vec::with_capacity(expanded_len);
        let mut media = Vec::with_capacity(images.len());
        for &token in tokens {
            if token != self.image {
                expanded.push(token);
                continue;
            }
            let index = media.len();
            let (h, w) = grids[index];
            media.push(MediaSpan { start: expanded.len(), len: images[index].tokens, key: images[index].key.into() });
            for _ in 0..h {
                expanded.extend(std::iter::repeat_n(self.image, w));
                expanded.push(self.row_break);
            }
            *expanded.last_mut().expect("nonempty span") = self.end;
        }
        Ok(ExpandedMediaPrompt { tokens: expanded, media })
    }
}
