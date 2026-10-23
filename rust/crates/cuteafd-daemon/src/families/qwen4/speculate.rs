//! Native MTP speculation for Qwen 3.8 Flash Next: the MTP's canonical
//! history, its draft chain, and verify-by-replay of the target.
//!
//! The MTP layer (vLLM `Qwen4ExpMultiTokenPredictor`) keeps its own K/V
//! records: position `p` holds the pair (target pre-mixer streams at `p`,
//! token `p + 1`), which predicts token `p + 2`. Target rows whose next token
//! is known wait in the engine's stash (`pending`, per state slot) until the
//! next draft cycle; a cycle's first MTP step runs every pending row (the
//! canonical history) and drafts from each sequence's last one; later steps
//! chain on the MTP's own output streams and the previous draft (positions
//! past the committed ones are rewritten by the next cycle's canonical rows).
//!
//! The target verifies `[next, d_0, .., d_{n-1}]` in one speculative step
//! (GDN and PLE record replay inputs instead of advancing); the accepted rows
//! are committed (`qwen4_gdn_commit` / `qwen4_ple_commit`), the placement and
//! n-gram history rewound, and the kept rows' streams stashed for the MTP.
use super::engine::{
    MtpGroup, MtpOut, MtpRow, MtpSource, MtpTokens, Qwen4Engine, Qwen4Placement, DECODE_ROWS, MTP_DEFERRED_STEPS,
    MTP_PENDING_ROWS,
};
use anyhow::{ensure, Result};
use cuteafd_loader::families::qwen4::NgramHistory;

/// A sequence's MTP bookkeeping: stash rows (position, token at position + 1)
/// whose pairs are not in the MTP's history yet.
#[derive(Debug, Clone, Default)]
pub(crate) struct MtpSeq {
    pub pending: Vec<(usize, u32)>,
    /// The last stash row's token is not known yet (the prompt's last row
    /// waits for the first sampled token).
    pub open: bool,
}

impl MtpSeq {
    /// The first sampled token completes the prompt's last pair.
    pub fn close(&mut self, token: u32) {
        if self.open {
            if let Some(last) = self.pending.last_mut() {
                last.1 = token;
            }
            self.open = false;
        }
    }
}

/// After the prefill of `chunk` (tokens from position `start`): the MTP's
/// canonical rows for every pair whose next token is known (`next`: the token
/// after the chunk, if known), and the chunk's last row stashed otherwise.
pub(crate) fn prefill_chunk(engine: &Qwen4Engine<'_>, placement: &Qwen4Placement, start: usize, chunk: &[u32], next: Option<u32>, seq: &mut MtpSeq) -> Result<()> {
    let n = chunk.len();
    ensure!(placement.mtp_frontier.get() == start, "MTP prefill requires contiguous canonical history");
    ensure!(n > 0 && seq.pending.is_empty(), "MTP prefill of an empty chunk or behind pending rows");
    let known = if next.is_some() { n } else { n - 1 };
    if next.is_none() {
        // Stash first: the MTP step reuses the target streams' buffer.
        engine.mtp_stash(false, &[(placement.slot, n - 1, 1, 0)])?;
        seq.pending.push((start + n - 1, 0));
        seq.open = true;
        placement.mtp_snapshot_pending.set(true);
    }
    if known == 0 {
        return Ok(());
    }
    let tokens: Vec<u32> = (0..known).map(|i| chunk.get(i + 1).copied().or(next).unwrap_or_default()).collect();
    for first in (0..known).step_by(engine.mtp_prefill_rows()) {
        let end = known.min(first + engine.mtp_prefill_rows());
        let rows = (first..end).map(|i| MtpRow { position: start + i, token: tokens[i], source: i as i32 }).collect();
        engine.mtp_step(false, &[MtpGroup { placement, rows }], MtpSource::Target, &[], MtpTokens::Host(&tokens[first..end]),
            MtpOut::Download { logits: false })?;
        placement.mtp_frontier.set(start + end);
    }
    Ok(())
}

/// One sequence's part of a draft cycle.
pub(crate) type DraftSeq<'p> = crate::shared::qwen::scheduler::DraftSeq<'p, Qwen4Placement, MtpSeq>;
pub(crate) use crate::shared::qwen::scheduler::DraftTiming;

impl crate::shared::qwen::scheduler::DraftState for MtpSeq {
    fn pending_rows(&self) -> usize { self.pending.len() }
    fn close(&mut self, token: u32) { MtpSeq::close(self, token); }
}

/// Runs the pending rows of every sequence (canonical history) and drafts up
/// to each sequence's depth; returns each sequence's drafts. Every chain
/// step embeds the previous step's drafts on the device, so the steps queue
/// back to back and the drafts come back once, at the end.
pub(crate) fn draft(engine: &Qwen4Engine<'_>, seqs: &mut [DraftSeq<'_>], timing: &mut DraftTiming)
    -> Result<Vec<Vec<u32>>> {
    let mut drafts = vec![Vec::new(); seqs.len()];
    let started = std::time::Instant::now();
    // Step 0: every pending row; heads on the last row of the drafting sequences.
    let mut groups = Vec::new();
    let mut heads = Vec::new();
    let mut head_of = vec![None; seqs.len()];
    let mut tokens = Vec::new();
    let mut last_position = vec![0usize; seqs.len()];
    let mut row = 0usize;
    for (i, s) in seqs.iter().enumerate() {
        if s.seq.pending.is_empty() {
            ensure!(s.depth == 0, "a drafting sequence has no pending MTP rows");
            continue;
        }
        ensure!(!s.seq.open, "a sequence drafts before its prompt's last pair is closed");
        let rows: Vec<MtpRow> = s.seq.pending.iter().enumerate()
            .map(|(j, &(position, token))| MtpRow { position, token, source: j as i32 }).collect();
        tokens.extend(rows.iter().map(|r| r.token));
        row += rows.len();
        last_position[i] = rows.last().map_or(0, |r| r.position);
        if s.depth > 0 {
            head_of[i] = Some(heads.len());
            heads.push(row - 1);
        }
        groups.push(MtpGroup { placement: s.placement, rows });
    }
    ensure!(row <= DECODE_ROWS, "{row} pending MTP rows exceed a decode step");
    if groups.is_empty() {
        return Ok(drafts);
    }
    let depth = seqs.iter().map(|s| s.depth).max().unwrap_or(0);
    ensure!(depth <= MTP_DEFERRED_STEPS, "MTP depth {depth}");
    engine.mtp_step(true, &groups, MtpSource::Pending, &heads, MtpTokens::Host(&tokens), MtpOut::Defer { step: 0 })?;
    timing.steps += 1;
    drop(groups);
    for s in seqs.iter_mut() {
        if let Some(&(position, _)) = s.seq.pending.last() {
            s.placement.mtp_frontier.set(position + 1);
        }
        s.seq.pending.clear();
    }
    // Per step, the sequences it drafted for, in draft order (step 0: by head).
    let mut steps: Vec<Vec<usize>> = Vec::new();
    let mut first = vec![usize::MAX; heads.len()];
    for (i, h) in head_of.iter().enumerate() {
        if let Some(h) = h {
            first[*h] = i;
        }
    }
    steps.push(first);
    // Chain steps: one row per sequence still drafting, reading its previous
    // head row and embedding its previous draft (still on the device).
    // Where each sequence's last draft sits: (step row, head row).
    let mut previous: Vec<Option<(usize, usize)>> = head_of.iter().map(|h| h.map(|h| (heads[h], h))).collect();
    for j in 1..depth {
        let mut groups = Vec::new();
        let mut index = Vec::new();
        let mut members = Vec::new();
        for (i, s) in seqs.iter().enumerate() {
            let Some((source, draft)) = previous[i] else { continue };
            if s.depth <= j {
                continue;
            }
            index.push(draft as u32);
            members.push(i);
            // The row's token is the previous draft (embedded on the device; the MTP's
            // history does not keep it).
            groups.push(MtpGroup { placement: s.placement, rows: vec![MtpRow { position: last_position[i] + j,
                token: 0, source: source as i32 }] });
        }
        if groups.is_empty() {
            break;
        }
        let heads: Vec<usize> = (0..groups.len()).collect();
        engine.mtp_step(true, &groups, MtpSource::Chain, &heads, MtpTokens::Drafts { step: j - 1, index: &index },
            MtpOut::Defer { step: j })?;
        timing.steps += 1;
        previous = vec![None; seqs.len()];
        for (r, &i) in members.iter().enumerate() {
            previous[i] = Some((r, r));
        }
        steps.push(members);
    }
    let counts: Vec<usize> = steps.iter().map(Vec::len).collect();
    for (step, ids) in steps.iter().zip(engine.mtp_drafts(&counts)?) {
        for (&i, id) in step.iter().zip(ids) {
            drafts[i].push(id);
        }
    }
    timing.seconds += started.elapsed().as_secs_f64();
    Ok(drafts)
}

/// A sequence's verified rows: the tokens it verified from `start` (next
/// token, then drafts), its n-gram history before the step, and its first
/// row in the step.
pub(crate) type Verified<'p> = crate::shared::qwen::scheduler::Verified<'p, Qwen4Placement, MtpSeq, NgramHistory>;

/// After a speculative verify: commits each sequence's kept rows to the GDN
/// and PLE state, rewinds its placement, and stashes the kept rows' streams
/// with their next tokens for the MTP. Finished retained turns normalize
/// canonical pending rows before snapshot capture.
pub(crate) fn accept(engine: &Qwen4Engine<'_>, verified: &mut [Verified<'_>], spec: bool) -> Result<()> {
    let mut commits = Vec::new();
    let mut stash = Vec::new();
    for v in verified.iter_mut() {
        // None means no retained frontier (cancelled/failed emission).
        let Some((kept, _after)) = v.kept else { continue };
        ensure!(kept >= 1 && kept <= v.rows.len(), "kept {kept} of {} rows", v.rows.len());
        if spec {
            commits.push((v.placement.slot, v.first_row, kept));
            engine.rewind(v.placement, v.start, v.history.clone(), &v.rows[..kept])?;
        } else {
            ensure!(kept == v.rows.len(), "a plain step keeps every row");
        }
        let neural_kept = v.neural_kept();
        if let Some(at) = stash_accepted(v.seq, v.start, v.rows, neural_kept, false)? {
            stash.push((v.placement.slot, v.first_row, kept, at));
        }
    }
    engine.commit(&commits)?;
    if !stash.is_empty() {
        engine.mtp_stash(true, &stash)?;
    }
    for row in verified.iter_mut().filter(|row| row.neural && row.finished && row.kept.is_some()) {
        normalize_snapshot(engine, row.placement, row.seq)?;
    }
    Ok(())
}

/// Consume every canonical pending row except hidden(P-1), then retain that
/// row verbatim. The next token is outside the snapshot and is filled on restore.
pub(crate) fn normalize_snapshot(engine: &Qwen4Engine<'_>, placement: &Qwen4Placement, seq: &mut MtpSeq) -> Result<()> {
    let n = seq.pending.len();
    ensure!(n > 0, "Qwen drafting snapshot has no final hidden row");
    if n > 1 {
        consume_pending(engine, placement, &seq.pending[..n - 1])?;
    }
    let position = seq.pending[n - 1].0;
    ensure!(position + 1 == placement.len && placement.mtp_frontier.get() == position,
        "Qwen snapshot pending row differs from canonical frontier");
    engine.mtp_keep_last_pending(placement.slot, n - 1)?;
    seq.pending.clear();
    seq.pending.push((position, 0));
    seq.open = true;
    placement.mtp_snapshot_pending.set(true);
    Ok(())
}

fn consume_pending(engine: &Qwen4Engine<'_>, placement: &Qwen4Placement, pending: &[(usize, u32)]) -> Result<()> {
    ensure!(!pending.is_empty() && pending[0].0 == placement.mtp_frontier.get()
        && pending.windows(2).all(|rows| rows[1].0 == rows[0].0 + 1), "noncontiguous canonical MTP stash");
    let rows = pending.iter().enumerate().map(|(source, &(position, token))|
        MtpRow { position, token, source: source as i32 }).collect();
    let tokens: Vec<_> = pending.iter().map(|&(_, token)| token).collect();
    engine.mtp_step(true, &[MtpGroup { placement, rows }], MtpSource::Pending, &[],
        MtpTokens::Host(&tokens), MtpOut::Download { logits: false })?;
    placement.mtp_frontier.set(pending.last().unwrap().0 + 1);
    Ok(())
}

pub(crate) fn restore_pending(engine: &Qwen4Engine<'_>, placement: &Qwen4Placement, frontier: usize,
    next: Option<u32>, seq: &mut MtpSeq) -> Result<()> {
    if frontier == 0 { return Ok(()); }
    if placement.mtp_frontier.get() == frontier { return Ok(()); }
    ensure!(placement.mtp_snapshot_pending.get() && placement.mtp_frontier.get() == frontier - 1,
        "Qwen restore is missing canonical MTP hidden row");
    seq.pending.push((frontier - 1, next.unwrap_or(0)));
    seq.open = next.is_none();
    if next.is_some() {
        consume_pending(engine, placement, &seq.pending)?;
        seq.pending.clear();
        placement.mtp_snapshot_pending.set(false);
    }
    Ok(())
}

fn stash_accepted(seq: &mut MtpSeq, start: usize, rows: &[u32], kept: Option<(usize, u32)>,
    finished: bool) -> Result<Option<usize>> {
    let Some((kept, after)) = kept.filter(|_| !finished) else { return Ok(None); };
    ensure!(kept > 0 && kept <= rows.len(), "invalid accepted MTP stash extent");
    let at = seq.pending.len();
    ensure!(at + kept <= MTP_PENDING_ROWS, "the MTP stash of a sequence overflows");
    for j in 0..kept {
        let next = if j + 1 < kept { rows[j + 1] } else { after };
        seq.pending.push((start + j, next));
    }
    Ok(Some(at))
}

#[cfg(test)]
mod request_state_tests {
    use super::*;

    #[test]
    fn mtp_cache_hit_and_interleaved_probe_keep_request_local_stashes() -> Result<()> {
        for start in [1usize, 63, 64, 65, 255, 256, 257, 511, 512, 513, 1492] {
            // Only predictor-complete marks are eligible for drafting;
            // accepted hidden rows remain local to the reader.
            let mut hit = MtpSeq::default();
            let mut cold = MtpSeq::default();
            let mut probe = MtpSeq { pending: vec![(0, 77)], open: false };
            let probe_before = probe.pending.clone();
            for position in start..start + 3 {
                let rows = [11, 13];
                let mut placements = [(), (), ()];
                let mut seqs = [&mut hit, &mut probe, &mut cold];
                for (index, (placement, seq)) in placements.iter_mut().zip(&mut seqs).enumerate() {
                    let v = crate::shared::qwen::scheduler::Verified { placement, seq: &mut **seq,
                        start: position, history: (), rows: &rows, first_row: 2 * index,
                        kept: Some((2, 17)), finished: false, neural: index != 1 };
                    let kept = v.neural_kept();
                    assert_eq!(stash_accepted(v.seq, v.start, v.rows, kept, v.finished)?.is_some(), index != 1);
                }
                assert_eq!(hit.pending, cold.pending);
                assert_eq!(probe.pending, probe_before);
                assert!(!probe.open);
                // The next draft consumes each request's own canonical rows.
                hit.pending.clear();
                cold.pending.clear();
            }
        }
        Ok(())
    }
}
