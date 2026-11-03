//! MiMo's native MTP drafter (`model.mtp.layers.0..2`; MiMo V2 Flash has no
//! other drafter): draft step `k` runs MTP layer `k` as one SWA decoder layer
//! with a dense MLP over `eh_proj(cat(enorm(embed(t)), hnorm(h)))`, then
//! `final_layernorm` and the target head.
//!
//! Stage `k` at row `j` takes the target's last-layer output `h_j` (pre-norm)
//! and the token `t_{j+k+1}` and predicts `t_{j+k+2}` (every stage reads the
//! target's hidden state: chaining a stage's own output measures far worse,
//! python/reference/families/mimo_v2/mimo_mtp/reference.py). Each stage keeps its own SWA ring
//! per sequence (the engine's ring layout and programs). A draft at a
//! sequence of length `len` (tokens `0..len` processed, `t_len` = next) runs
//! stage `k` over its rows `ext[k]..len`: rows `j <= len - k - 1` with their
//! true tokens (committed to the ring for good), rows past with the drafts
//! `d_1..d_k` (committed too, recomputed by the next draft once true). The
//! draft is the argmax of row `len - 1`.
//!
//! The engine taps every step's last-layer rows into a hidden ring (256
//! positions per sequence ring) that the stages read.
use crate::shared::memory::DeviceAllocation;
use super::weights::MimoLayer;

pub(crate) type Dev<'a> = DeviceAllocation<'a>;

/// Positions of target hidden rows kept per sequence ring.
pub(crate) const HIDDEN_ROWS: usize = 256;

/// One MTP stage: its SWA decoder layer, extra weights and per-ring SWA records.
pub(crate) struct MtpStage<'a> {
    pub layer: MimoLayer<'a>,
    /// `eh_proj` BF16 [H, 2H], `enorm`, `hnorm`, `final_layernorm`.
    pub eh: Dev<'a>,
    pub enorm: Dev<'a>,
    pub hnorm: Dev<'a>,
    pub final_norm: Dev<'a>,
    /// [rings * 256, record] SWA records.
    pub ring: Dev<'a>,
}

pub(crate) struct MtpDrafter<'a> {
    pub projection: cuteafd_ffi::draft_projection::DraftProjections<'a>,
    pub stages: Vec<MtpStage<'a>>,
    /// [rings * HIDDEN_ROWS, H] BF16: target last-layer outputs by (ring, position % 256).
    pub hidden: Dev<'a>,
    /// Scratch [DECODE_ROWS, H] x 3 and [DECODE_ROWS, 2H].
    pub embed: Dev<'a>,
    pub rows_h: Dev<'a>,
    pub normed_e: Dev<'a>,
    pub normed_h: Dev<'a>,
    pub cat: Dev<'a>,
    /// Per ring and stage: rows `0..ext` of the stage's ring hold true tokens.
    pub ext: std::cell::RefCell<Vec<Vec<usize>>>,
    /// Token ids of a draft cycle (U32): a pass's known tokens (DECODE_ROWS),
    /// then stage `k`'s draft of member `i` at `(1 + k) * DECODE_ROWS + i`.
    pub ids: Dev<'a>,
    /// A pass's row -> `ids` index (U32 [DECODE_ROWS]).
    pub index: Dev<'a>,
}

/// The token a pass row embeds: known on the host, or an earlier stage's
/// draft (still on the device) of the cycle's member `member`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Token {
    Known(u32),
    Draft { stage: usize, member: usize },
}

/// One sequence's MTP draft request: its ring, its length (`t_len` next) and
/// its tokens `0..=len` (prompt, generated and the next token).
#[derive(Debug, Clone)]
pub(crate) struct MtpSeq<'t> {
    pub ring: usize,
    pub len: usize,
    pub tokens: &'t [u32],
    pub media: Option<&'t cuteafd_engine::media::RequestMedia>,
}

/// Known MTP inputs are shifted by stage+1 relative to target hidden rows.
pub(crate) fn embedding_media(seqs: &[MtpSeq<'_>], stage: usize,
    groups: &[(usize, usize, Vec<Token>)]) -> anyhow::Result<cuteafd_engine::media::MediaChunk> {
    let mut packed = cuteafd_engine::media::MediaChunk::default();
    let mut offset = 0;
    for (ring, first, tokens) in groups {
        if let Some(media) = seqs.iter().find(|s| s.ring == *ring).and_then(|s| s.media) {
            let mut chunk = cuteafd_engine::media::MediaChunk::default();
            let start = first + stage + 1;
            media.write_chunk(start, start + tokens.len(), &mut chunk)?;
            for index in chunk.indices {
                anyhow::ensure!(matches!(tokens[index as usize], Token::Known(_)), "image row cannot be a draft");
                packed.indices.push(u32::try_from(offset + index as usize)?);
            }
            packed.features.extend(chunk.features);
        }
        offset += tokens.len();
    }
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_engine::media::{EmbeddingCache, ImageKey, MediaSpan, RequestMedia};
    use std::sync::Arc;

    #[test]
    fn mtp_media_tracks_shifted_known_rows_across_rings() {
        let key = ImageKey([5; 32]);
        let mut cache = EmbeddingCache::new(12);
        let pin = cache.reserve(key, 12).unwrap();
        let lease = cache.complete(key, Arc::from((0u8..12).collect::<Vec<_>>())).unwrap();
        let mut media = RequestMedia::new(vec![MediaSpan { start: 3, len: 3, key: key.into() }], 2, 8).unwrap();
        media.attach(lease).unwrap();
        drop(pin);
        let seqs = [MtpSeq { ring: 1, len: 8, tokens: &[], media: Some(&media) },
            MtpSeq { ring: 0, len: 8, tokens: &[], media: None }];
        let groups = [(0, 0, vec![Token::Known(1); 2]), (1, 2, vec![Token::Known(2); 3])];
        let chunk = embedding_media(&seqs, 1, &groups).unwrap();
        assert_eq!(chunk.indices, [2, 3]);
        assert_eq!(chunk.features, (4u8..12).collect::<Vec<_>>());
        let draft = [(1, 1, vec![Token::Draft { stage: 0, member: 0 }])];
        assert!(embedding_media(&seqs, 1, &draft).is_err());
    }
}
