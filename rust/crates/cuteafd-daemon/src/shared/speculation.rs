//! Shared speculative draft helpers; serving lifecycles stay with each family.

pub(crate) mod block_dspark;
pub(crate) mod recurrent_weights;
pub(crate) mod recurrent_forward;
pub(crate) mod recurrent_head;
pub(crate) mod recurrent_context;
pub(crate) mod recurrent_step;
pub(crate) mod recurrent_cache;
pub(crate) mod recurrent_tp2;
pub(crate) mod recurrent_snapshots;
pub(crate) mod recurrent_pages;
pub(crate) mod recurrent;
#[cfg(test)]
pub(crate) mod recurrent_oracle;
pub(crate) mod dflash;
pub(crate) mod gqa_tp2;
pub(crate) mod target;

pub(crate) mod weights;

/// Retain the original ingest chunk if any of it survives at `end`. Never trim
/// a straddling chunk: projection rounding can depend on its GEMM row count.
/// Context K/V are independent projections of each position's target taps, so
/// chunks overwritten before the first draft need neither capture nor update.
pub(crate) fn retain_context_chunk(window: Option<usize>, positions: std::ops::Range<usize>, end: usize) -> bool {
    !positions.is_empty() && window.is_none_or(|window| positions.end > end.saturating_sub(window))
}

/// Planned marks split the discard horizon before the final prompt frontier.
pub(crate) fn next_context_frontier(prompt_end: usize, points: &[usize], start: usize) -> usize {
    points.iter().copied().filter(|&point| point > start).min().unwrap_or(prompt_end).min(prompt_end)
}

/// The latest skipped position is dead only once a full window follows it.
pub(crate) fn context_window_complete(window: usize, skipped_end: usize, frontier: usize) -> bool {
    skipped_end <= frontier.saturating_sub(window)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_families_route_the_declared_score_sources() {
        // These are wiring checks for adapters whose constructors require an
        // admitted accelerator. The policy itself is exercised with CPU replays.
        for (family, source, declaration) in [
            ("GLM 5.3 / ML4", include_str!("../families/glm5/serve.rs"), "d.score_source(), true"),
            ("GLM Flash", include_str!("../families/glm5_flash/serve.rs"), "drafter.head_kind(), true"),
            ("GLM Flash routes", include_str!("../families/glm5_flash/dspark.rs"), "Self::Dflash2(d) => d.score_source()"),
            ("Kimi", include_str!("../families/kimi_k3/draft_common.rs"), "Self::Block { .. } => cuteafd_core::HeadScore::Probability"),
            ("Kimi DFlash2", include_str!("../families/kimi_k3/draft_common.rs"), "Self::Mla(d) => d.score_source()"),
            ("Muse", include_str!("../families/muse_spark/backend.rs"), "|d| d.drafter.score_source()"),
            ("V4.1", include_str!("../families/deepseek_v41/v41_native_serve/speculative.rs"), "cuteafd_core::HeadScore::Logit"),
            ("V4.1 DFly2", include_str!("../families/deepseek_v41/v41_native_serve/speculative.rs"), "if self.is_dfly2() {\n            cuteafd_core::HeadScore::CandidateProbability"),
            ("MiniMax", include_str!("../families/minimax_m3/backend.rs"), "HeadScore { cuteafd_core::HeadScore::Probability }"),
            ("Qwen dense", include_str!("../families/qwen3_5/serve.rs"), "self.drafter.is_some() { cuteafd_core::HeadScore::Probability }"),
            ("DeepSeek V4", include_str!("../families/deepseek_v4/serve.rs"), "engine.draft_block(), cuteafd_core::HeadScore::None"),
            ("Inkling", include_str!("../families/inkling2/backend.rs"), "draft.is_some() { cuteafd_core::HeadScore::CandidateProbability }"),
            ("Qwen Next", include_str!("../families/qwen4/engine.rs"), "head: HeadScore::None"),
        ] {
            assert!(source.contains(declaration), "{family} score source");
        }
        let flash = include_str!("../families/glm5_flash/dspark.rs");
        assert!(flash.contains("Self::Dspark(_) => cuteafd_core::HeadScore::Probability"));
        assert!(flash.contains("Self::RemoteDflash(d) => d.draft.score_source()"));
        let selector = include_str!("speculation/dflash.rs");
        assert!(selector.contains("DraftHeadOps::Selector { .. } => cuteafd_core::HeadScore::CandidateProbability"));
        assert!(selector.contains("DraftHeadOps::RnnMarkov { .. } | DraftHeadOps::Peer => cuteafd_core::HeadScore::None"));
    }

    #[test]
    fn planned_marks_and_disconnects_publish_only_complete_windows() {
        let (window, end) = (4096, 16384);
        let points = [8192, 12288];
        let (mut full, mut candidate) = (vec![0usize; window], vec![0usize; window]);
        let mut skipped_end = 0;
        let mut withheld = 0;
        let mut retained = 0;
        for start in (0..end).step_by(128) {
            let frontier = next_context_frontier(end, &points, start);
            let keep = retain_context_chunk(Some(window), start..start + 128, frontier);
            if !keep { skipped_end = start + 128; }
            for position in start..start + 128 {
                full[position % window] = position + 1;
                if keep { candidate[position % window] = position + 1; }
            }
            // An unexpected disconnect may park only after all holes died.
            if context_window_complete(window, skipped_end, start + 128) {
                assert_eq!(full, candidate);
                retained += 1;
            } else { withheld += 1; }
            if points.contains(&(start + 128)) || start + 128 == end {
                assert!(context_window_complete(window, skipped_end, start + 128));
                assert_eq!(full, candidate, "planned prompt/intermediate mark");
                let restored = candidate.clone();
                assert_eq!(restored, full, "restored mark equals cold prefill");
            }
        }
        assert!(withheld > 0 && retained > 0);
        // A turn continues from the exact prompt ring without a new discard horizon.
        for position in end..end + 17 {
            full[position % window] = position + 1;
            candidate[position % window] = position + 1;
        }
        assert_eq!(full, candidate, "turn-end mark");
    }

    #[test]
    fn flash_serial_packed_and_replay_keep_original_chunks_and_ring_bytes() {
        let (window, prompt_end) = (2048, 8192);
        // Flash serial calls keep the last TAP_ROWS, packed calls use each
        // segment's tap offset, and replay ingests TAP_ROWS from its origin.
        for (origin, chunk, tap_offset) in [(0, 2048, 0), (137, 511, 79), (0, 257, 0)] {
            let mut full = vec![0u64; window];
            let mut candidate = full.clone();
            let mut position = origin;
            while position < prompt_end {
                let count = chunk.min(prompt_end - position);
                let keep = retain_context_chunk(Some(window), position..position + count, prompt_end);
                for row in 0..count {
                    let payload = ((position + row + 1) as u64) << 32
                        | ((count as u64) << 16) | (tap_offset + row) as u64;
                    full[(position + row) % window] = payload;
                    if keep { candidate[(position + row) % window] = payload; }
                }
                position += count;
            }
            assert_eq!(full, candidate, "origin={origin} chunk={chunk} tap_offset={tap_offset}");
        }
        let flash = include_str!("../families/glm5_flash/serve.rs");
        assert!(flash.contains("prefill_packed_draft_ends(&mut sequences, &draft_ends"));
        assert!(flash.contains("drafter.retain_context_chunk(first..first + n, draft_end)"));
        assert!(flash.contains("drafter.retain_context_chunk(end - n..end, draft_context_end(p, p.done))"));
        let replay = include_str!("speculation/dflash.rs");
        assert!(replay.contains("retain_context_chunk(drafter.context_window(), done..done + n, p)"));
    }

    #[test]
    fn whole_chunk_ingest_matches_full_ring_and_preserves_shapes() {
        let window = 4096;
        for end in [17, window, window + 1, 8192, 65536] {
            for chunk in [128, 257, 2048] {
                let mut full = vec![[0u8; 16]; window];
                let mut skipped = full.clone();
                let mut kept = 0;
                for first in (0..end).step_by(chunk) {
                    let span = first..(first + chunk).min(end);
                    let retain = retain_context_chunk(Some(window), span.clone(), end);
                    for position in span.clone() {
                        // Include shape in the payload to detect accidental boundary trimming.
                        let mut bytes = [0; 16];
                        bytes[..8].copy_from_slice(&(position as u64).to_le_bytes());
                        bytes[8..].copy_from_slice(&(span.len() as u64).to_le_bytes());
                        full[position % window] = bytes;
                        if retain { skipped[position % window] = bytes; }
                    }
                    if retain { kept += span.len(); }
                }
                assert_eq!(full, skipped, "end={end} chunk={chunk}");
                assert!(kept <= end.min(window + chunk - 1));
            }
        }
        assert!(!retain_context_chunk(Some(window), 0..128, 8192));
        assert!(retain_context_chunk(Some(window), 4096..4224, 8193));
        assert!(retain_context_chunk(None, 0..128, 65536));
    }
}
