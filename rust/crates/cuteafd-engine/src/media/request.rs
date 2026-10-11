use super::{keys::validate_spans, EmbeddingLease, MediaKey, MediaError, MediaSpan};

/// Lazy per-span BF16 features. Prefix restores need only the span descriptors.
#[derive(Clone, Debug)]
pub struct RequestMedia {
    spans: Vec<MediaSpan>,
    features: Vec<Option<EmbeddingLease>>,
    row_bytes: usize,
    prompt_len: usize,
}
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MediaChunk {
    /// Chunk-relative rows overwritten by the injection kernel, in packed-feature order.
    pub indices: Vec<u32>,
    pub features: Vec<u8>,
    pub mask: Vec<u8>,
}
impl RequestMedia {
    pub fn new(
        spans: Vec<MediaSpan>,
        hidden_width: usize,
        prompt_len: usize,
    ) -> Result<Self, MediaError> {
        validate_spans(&spans)?;
        let row_bytes = hidden_width
            .checked_mul(2)
            .filter(|&n| n > 0)
            .ok_or(MediaError::Features)?;
        for span in &spans {
            if span.checked_end().unwrap() > prompt_len
                || span.len.checked_mul(row_bytes).is_none()
                || spans.iter().any(|s| s.key == span.key && s.len != span.len)
            {
                return Err(MediaError::Spans);
            }
        }
        let features = vec![None; spans.len()];
        Ok(Self {
            spans,
            features,
            row_bytes,
            prompt_len,
        })
    }
    pub fn spans(&self) -> &[MediaSpan] {
        &self.spans
    }
    pub fn span_features(&self, index: usize) -> Option<&std::sync::Arc<[u8]>> {
        self.features.get(index)?.as_ref()?.features()
    }
    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }
    pub fn needed(&self, resume: usize, end: usize) -> impl Iterator<Item = &MediaSpan> {
        self.spans
            .iter()
            .filter(move |s| s.start < end && s.checked_end().unwrap() > resume)
    }
    pub fn ready(&self, resume: usize, end: usize) -> bool {
        self.spans.iter().zip(&self.features).all(|(s, f)| {
            s.start >= end
                || s.checked_end().unwrap() <= resume
                || f.as_ref().is_some_and(|f| f.features().is_some())
        })
    }
    pub fn attach(&mut self, lease: EmbeddingLease) -> Result<(), MediaError> {
        let rows = lease.features().ok_or(MediaError::NotReady(lease.key()))?;
        let mut found = false;
        for (span, slot) in self.spans.iter().zip(&mut self.features) {
            if span.key == lease.key() {
                if rows.len() != span.len * self.row_bytes {
                    return Err(MediaError::Features);
                }
                *slot = Some(lease.clone());
                found = true;
            }
        }
        if !found {
            return Err(MediaError::Spans);
        }
        Ok(())
    }
    /// Probe-only override: bypasses the encoder with an admitted, domain-separated
    /// feature lease. The real span identity stays unchanged; callers must cold-admit
    /// the probe and disable every prefix snapshot. Never cache overrides by image key.
    pub fn attach_probe_override(&mut self, image_key: impl Into<MediaKey>, lease: EmbeddingLease) -> Result<(), MediaError> {
        let image_key = image_key.into();
        if lease.key() == image_key { return Err(MediaError::Features); }
        let rows = lease.features().ok_or(MediaError::NotReady(lease.key()))?;
        let mut found = false;
        for (span, slot) in self.spans.iter().zip(&mut self.features) {
            if span.key == image_key {
                if rows.len() != span.len * self.row_bytes { return Err(MediaError::Features); }
                *slot = Some(lease.clone());
                found = true;
            }
        }
        if !found { return Err(MediaError::Spans); }
        Ok(())
    }
    pub fn has_features(&self, key: impl Into<MediaKey>) -> bool {
        let key = key.into();
        self.spans
            .iter()
            .zip(&self.features)
            .any(|(s, f)| s.key == key && f.is_some())
    }
    /// Reuse a caller-owned staging vector. Decode chunks outside the prompt have no media.
    pub fn write_chunk(
        &self,
        start: usize,
        end: usize,
        out: &mut MediaChunk,
    ) -> Result<(), MediaError> {
        let count = end
            .checked_sub(start)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or(MediaError::Features)?;
        if !self.ready(start, end) {
            let key = self
                .spans
                .iter()
                .zip(&self.features)
                .find(|(s, f)| s.start < end && s.checked_end().unwrap() > start && f.is_none())
                .unwrap()
                .0
                .key;
            return Err(MediaError::NotReady(key));
        }
        out.indices.clear();
        out.features.clear();
        out.mask.clear();
        out.mask.resize(count, 0);
        for (span, lease) in self.spans.iter().zip(&self.features) {
            let lo = start.max(span.start);
            let hi = end.min(span.checked_end().unwrap());
            if lo >= hi {
                continue;
            }
            let data = lease.as_ref().unwrap().features().unwrap();
            out.features.extend_from_slice(
                &data[(lo - span.start) * self.row_bytes..(hi - span.start) * self.row_bytes],
            );
            for row in lo..hi {
                out.indices.push((row - start) as u32);
                out.mask[row - start] = 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ImageKey;
    use crate::media::EmbeddingCache;
    use std::sync::Arc;
    #[test]
    fn span_features_borrows_attached_features_by_index() {
        let key = ImageKey([7; 32]);
        let other = ImageKey([8; 32]);
        let mut request = RequestMedia::new(vec![
            MediaSpan { start: 0, len: 1, key: key.into() },
            MediaSpan { start: 1, len: 1, key: other.into() },
            MediaSpan { start: 2, len: 1, key: key.into() },
        ], 2, 3).unwrap();
        assert!(request.span_features(0).is_none());
        assert!(request.span_features(3).is_none());
        let features: Arc<[u8]> = Arc::from([0, 0, 0x80, 0x3f]);
        let mut cache = EmbeddingCache::new(4);
        let pin = cache.reserve(key, 4).unwrap();
        let lease = cache.complete(key, Arc::clone(&features)).unwrap();
        request.attach(lease).unwrap();
        drop(pin);
        assert!(Arc::ptr_eq(request.span_features(0).unwrap(), &features));
        assert!(request.span_features(1).is_none());
        assert!(Arc::ptr_eq(request.span_features(2).unwrap(), &features));
        assert!(request.span_features(usize::MAX).is_none());
    }
    #[test]
    fn probe_override_uses_a_separate_budgeted_cache_identity() {
        let image = ImageKey([7; 32]); let override_key = ImageKey([8; 32]);
        let mut request = RequestMedia::new(vec![MediaSpan { start: 1, len: 1, key: image.into() }], 2, 3).unwrap();
        let mut cache = EmbeddingCache::new(4);
        let pin = cache.reserve(override_key, 4).unwrap();
        let lease = cache.complete(override_key, Arc::from([0, 0, 0x80, 0x3f])).unwrap();
        request.attach_probe_override(image, lease).unwrap(); drop(pin);
        assert_eq!(request.spans()[0].key, image.into());
        assert!(!cache.contains(image)); assert!(cache.contains(override_key));
        assert!(cache.reserve(image, 4).is_err(), "override remains admitted and pinned");
        let mut chunk = MediaChunk::default(); request.write_chunk(0, 3, &mut chunk).unwrap();
        assert_eq!(chunk.indices, [1]); assert_eq!(chunk.features, [0, 0, 0x80, 0x3f]);
        let mut native = EmbeddingCache::new(4); let pin = native.reserve(image, 4).unwrap();
        let lease = native.complete(image, Arc::from([0; 4])).unwrap();
        assert!(request.attach_probe_override(image, lease).is_err()); drop(pin);
    }
    #[test]
    fn lazy_spans_stage_only_overlapping_rows_with_native_row_indices() {
        let key = ImageKey([7; 32]);
        let span = MediaSpan {
            start: 2,
            len: 3,
            key: key.into(),
        };
        let mut request = RequestMedia::new(vec![span], 2, 6).unwrap();
        assert!(request.ready(5, 6));
        assert!(!request.ready(4, 6));
        assert_eq!(request.needed(5, 6).count(), 0);
        let mut out = MediaChunk::default();
        assert!(request.write_chunk(1, 4, &mut out).is_err());
        let mut cache = EmbeddingCache::new(12);
        let reserved = cache.reserve(key, 12).unwrap();
        let lease = cache
            .complete(key, Arc::from((0u8..12).collect::<Vec<_>>()))
            .unwrap();
        request.attach(lease).unwrap();
        drop(reserved);
        request.write_chunk(3, 6, &mut out).unwrap();
        assert_eq!(out.indices, [0, 1]);
        assert_eq!(out.features, (4u8..12).collect::<Vec<_>>());
        assert_eq!(out.mask, [1, 1, 0]);
        request.write_chunk(6, 8, &mut out).unwrap();
        assert!(out.features.is_empty() && out.indices.is_empty());
        assert_eq!(out.mask, [0, 0]);
    }
}
