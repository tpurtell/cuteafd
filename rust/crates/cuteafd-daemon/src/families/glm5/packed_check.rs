//! Stored-token packed/singleton exact or fidelity gate, including a resumed prompt.
use super::engine::{GlmEngine, GlmPlacement, PageAllocator, PAGE_ROWS, RECORD_BYTES, INDEX_PAGE_BYTES};
use crate::shared::prefill_packing::{self, Limits};
use crate::shared::spark_intake::SparkLink;
use anyhow::{ensure, Context, Result};

const LENGTHS: [usize; 10] = [29, 40, 7, 64, 65, 127, 1, 255, 384, 100];

fn sequence(tokens: &[u32], member: usize, rows: usize) -> Vec<u32> {
    (0..rows).map(|row| tokens[(member * 97 + row) % tokens.len()]).collect()
}

fn state(engine: &GlmEngine<'_>, placement: &GlmPlacement) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for rank in 0..engine.ranks() {
        engine.on(rank, || {
            // SAFETY: the diagnostic owns this engine; all reads follow its queued work.
            unsafe { engine.library.cuda_stream_synchronize(engine.stream_of(rank))?; }
            for (records, index) in engine.paged_buffers_on(rank) {
                for (logical, &page) in placement.pages.iter().enumerate().take(placement.len.div_ceil(PAGE_ROWS)) {
                    if engine.context_kv() && page as usize % 2 != rank { continue; }
                    let local = if engine.context_kv() { page as usize / 2 } else { page as usize };
                    let rows = (placement.len - logical * PAGE_ROWS).min(PAGE_ROWS);
                    let mut read = |buffer, offset, bytes| -> Result<()> {
                        let view = crate::shared::prefix::view(buffer, offset, bytes)?;
                        let first = out.len();
                        out.resize(first + bytes, 0);
                        engine.library.copy_d2h(&mut out[first..], view)?;
                        Ok(())
                    };
                    if let Some((records, row_bytes)) = records {
                        read(records, local * PAGE_ROWS * row_bytes, rows * row_bytes)?;
                    }
                    if let Some(index) = index {
                        read(index, local * INDEX_PAGE_BYTES, rows * 128)?;
                        read(index, local * INDEX_PAGE_BYTES + PAGE_ROWS * 128, rows * 4)?;
                    }
                }
            }
            Ok(())
        })?;
    }
    Ok(out)
}

pub(crate) fn check(engine: &GlmEngine<'_>, tokens: &[u32], golden: &std::path::Path, members: usize,
    mode: prefill_packing::PackingMode, mut transport: Option<&mut SparkLink<'_>>,
    runtime: &tokio::runtime::Runtime) -> Result<()> {
    ensure!(!tokens.is_empty() && (2..=LENGTHS.len()).contains(&members), "--packed-check needs 2..10 members and stored tokens");
    let resumed = members - 1;
    let prefix = |i| if i == resumed { 2100 } else { 0 };
    let lengths = LENGTHS[..members - 1].iter().copied().chain([100]).collect::<Vec<_>>();
    let chunks = lengths.iter().enumerate().map(|(i, &rows)| (prefix(i), rows)).collect::<Vec<_>>();
    let limits = engine.packed_limits();
    for chunk in &chunks {
        prefill_packing::plan(&[*chunk], &Limits { rows: limits.chunk_rows, ..limits })?;
    }
    let groups = prefill_packing::groups(members, |range| prefill_packing::plan(&chunks[range], &limits).is_ok());
    let exact = mode == prefill_packing::PackingMode::Exact;
    let fallback = exact && limits.rows == 0;
    ensure!(fallback || groups.iter().any(|group| group.len() > 1),
        "packed-check must exercise a pack when packing is enabled");
    if fallback {
        println!("packed check: exact singleton fallback only, NOT a packing gate: {}", engine.packed_fallback_reason());
    } else if exact && members >= 6 && limits.rows <= 64 {
        ensure!(!groups.iter().any(|group| group.contains(&3) && group.contains(&4)),
            "exact packed group crosses the adjacent 64/65-row owner boundary");
        println!("packed check: exact cap {}, adjacent 64/65-row members split", limits.rows);
    }
    let prompts = lengths.iter().enumerate().map(|(i, &rows)| sequence(tokens, i, prefix(i) + rows)).collect::<Vec<_>>();
    let mut allocator = PageAllocator::new(engine);
    let mut packed = prompts.iter().map(|p| allocator.admit(p.len() + 1)).collect::<Result<Vec<_>>>()?;
    let mut alone = prompts.iter().map(|p| allocator.admit(p.len() + 1)).collect::<Result<Vec<_>>>()?;
    for placement in [&mut packed[resumed], &mut alone[resumed]] {
        for chunk in prompts[resumed][..2100].chunks(engine.prefill_capacity()) {
            engine.prefill_device(placement, chunk, transport.as_deref_mut().map(|t| (t, runtime)), 0)?;
        }
    }
    let chunk = |i: usize| &prompts[i][prefix(i)..];
    let mut packed_logits = vec![Vec::new(); members];
    let mut packed_groups = 0;
    for range in &groups {
        if range.len() == 1 {
            let i = range.start;
            packed_logits[i] = engine.prefill_device(&mut packed[i], chunk(i),
                transport.as_deref_mut().map(|t| (t, runtime)), 1)?.context("packed-check head")?
                .row_host(engine.library, 0)?;
        } else {
            packed_groups += 1;
            let first = range.start;
            let mut sequences = packed[range.clone()].iter_mut().enumerate().map(|(i, p)| (p, chunk(first + i))).collect::<Vec<_>>();
            let (_, outcomes) = engine.prefill_packed(&mut sequences, &mut |i, logits| {
                packed_logits[first + i] = logits.row_host(engine.library, 0)?;
                Ok(Ok(()))
            })?;
            for outcome in outcomes { outcome?; }
        }
    }
    let (mut equal, mut top1, mut worst_kl) = (true, 0usize, 0.0f64);
    for (i, placement) in alone.iter_mut().enumerate() {
        let logits = engine.prefill_device(placement, chunk(i), transport.as_deref_mut().map(|t| (t, runtime)), 1)?
            .context("packed-check singleton head")?.row_host(engine.library, 0)?;
        let same_logits = packed_logits[i].iter().map(|f| f.to_bits()).eq(logits.iter().map(|f| f.to_bits()));
        let same_state = state(engine, &packed[i])? == state(engine, placement)?;
        let kl = super::mean_kl(&packed_logits[i], &logits, logits.len());
        let argmax = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(id, _)| id);
        let same_top1 = argmax(&packed_logits[i]) == argmax(&logits);
        top1 += usize::from(same_top1);
        worst_kl = worst_kl.max(kl);
        ensure!(kl.is_finite(), "packed sequence {i} produced nonfinite KL");
        println!("packed check: sequence {i}: {} rows at {}: logits {}, live cache {}, KL {kl:.8}, top-1 {same_top1}",
            lengths[i], prefix(i), if same_logits { "identical" } else { "DIFFER" },
            if same_state { "identical" } else { "DIFFER" });
        equal &= same_logits && same_state;
    }
    for placement in packed.into_iter().chain(alone) { allocator.release(placement); }
    println!("packed check: {members} sequences, {packed_groups} packed groups, {} singleton groups; {}",
        groups.len() - packed_groups, if equal { "every sequence identical" } else { "DIFFER" });
    if exact {
        ensure!(equal, "exact prefill differs from its singleton reference");
    } else {
        let agreement = top1 as f64 / members as f64;
        println!("packed versus singleton (informational): worst member KL {worst_kl:.8}, top-1 {agreement:.4}");
        golden_fidelity(engine, tokens, golden, members, transport, runtime)?;
    }
    Ok(())
}

#[derive(Default)]
struct Fidelity {
    rows: usize,
    kl_sum: f64,
    top1: usize,
}

impl Fidelity {
    fn add(&mut self, logits: &[f32], golden: &[f32], vocab: usize) -> Result<()> {
        ensure!(vocab > 0 && logits.len() == golden.len() && logits.len().is_multiple_of(vocab),
            "packed golden logits shape differs");
        ensure!(logits.iter().chain(golden).all(|v| v.is_finite()), "nonfinite packed golden logits");
        let rows = logits.len() / vocab;
        self.kl_sum += super::mean_kl(logits, golden, vocab) * rows as f64;
        self.top1 += logits.chunks_exact(vocab).zip(golden.chunks_exact(vocab))
            .filter(|(a, b)| super::argmax(a) == super::argmax(b)).count();
        self.rows += rows;
        Ok(())
    }

    fn metrics(&self) -> Result<(f64, f64)> {
        ensure!(self.rows > 0 && self.kl_sum.is_finite(), "empty or nonfinite fidelity score");
        Ok((self.kl_sum / self.rows as f64, self.top1 as f64 / self.rows as f64))
    }

    fn non_inferior(&self, control: &Self) -> Result<(f64, f64)> {
        ensure!(self.rows == control.rows, "packed and singleton fidelity rows differ");
        let (kl, top1) = self.metrics()?;
        let (base_kl, base_top1) = control.metrics()?;
        let delta = (kl - base_kl, top1 - base_top1);
        ensure!(delta.0 <= 0.005 + 1e-12 && delta.1 >= -0.01 - 1e-12,
            "unrestricted packing inferior to singleton: delta KL {:.8}, delta top-1 {:.4}; bars +0.005/-0.01",
            delta.0, delta.1);
        Ok(delta)
    }

    fn pool(scores: &[Self]) -> Self {
        scores.iter().fold(Self::default(), |mut total, score| {
            total.rows += score.rows;
            total.kl_sum += score.kl_sum;
            total.top1 += score.top1;
            total
        })
    }
}

/// Stored reference rows are valid only for the original, unshifted token prefix.
/// Keep the independent shifted/wrapped prompts above for isolation/exact checks.
fn golden_fidelity(engine: &GlmEngine<'_>, tokens: &[u32], golden: &std::path::Path,
    members: usize, mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    use sha2::{Digest, Sha256};
    let resumed = members - 1;
    let start = |i| if i == resumed { 1024 } else { 0 };
    let lengths = LENGTHS[..resumed].iter().copied().chain([100]).collect::<Vec<_>>();
    ensure!(lengths.iter().enumerate().all(|(i, &rows)| start(i) + rows <= tokens.len()),
        "stored golden needs every unshifted prompt prefix");
    let vocab = engine.cfg.vocab_size;
    let mut reference = std::fs::File::open(golden.join("logits.bin"))?;
    ensure!(reference.metadata()?.len() == (tokens.len() * vocab * 4) as u64, "stored golden logits shape differs");
    let mut allocator = PageAllocator::new(engine);
    let mut placements = lengths.iter().enumerate().map(|(i, &rows)| allocator.admit(start(i) + rows + 1))
        .collect::<Result<Vec<_>>>()?;
    for part in tokens[..start(resumed)].chunks(engine.prefill_capacity()) {
        engine.prefill_device(&mut placements[resumed], part,
            transport.as_deref_mut().map(|t| (t, runtime)), 0)?;
    }
    let limits = engine.packed_limits();
    let chunks = lengths.iter().enumerate().map(|(i, &rows)| (start(i), rows)).collect::<Vec<_>>();
    let groups = prefill_packing::groups(members, |range| prefill_packing::plan(&chunks[range], &limits).is_ok());
    ensure!(groups.iter().any(|g| g.len() > 1), "stored-golden fidelity must exercise a real pack");
    let mut scores = (0..members).map(|_| Fidelity::default()).collect::<Vec<_>>();
    let mut score_segment = |i: usize, segment, score: &mut Fidelity| -> Result<Vec<[u8; 32]>> {
        let mut digests = Vec::new();
        engine.packed_check_logits(segment, &mut |offset, logits| {
            let logits = logits.to_host(engine.library)?;
            let bytes = logits.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect::<Vec<_>>();
            digests.push(Sha256::digest(&bytes).into());
            let mut bytes = vec![0u8; logits.len() * 4];
            reference.seek(SeekFrom::Start(((start(i) + offset) * vocab * 4) as u64))?;
            reference.read_exact(&mut bytes)?;
            let golden = bytes.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
            score.add(&logits, &golden, vocab)
        })?;
        Ok(digests)
    };
    let mut packed_digests = vec![Vec::new(); members];
    for range in groups {
        let first = range.start;
        let mut sequences = placements[range.clone()].iter_mut().enumerate().map(|(j, p)| {
            let i = first + j;
            (p, &tokens[start(i)..start(i) + lengths[i]])
        }).collect::<Vec<_>>();
        let (segments, outcomes) = engine.prefill_packed(&mut sequences, &mut |_, _| Ok(Ok(())))?;
        for outcome in outcomes { outcome?; }
        for (j, segment) in segments.into_iter().enumerate() {
            let i = first + j;
            packed_digests[i] = score_segment(i, segment, &mut scores[i])?;
        }
    }
    for p in placements { allocator.release(p); }
    let mut controls = (0..members).map(|_| Fidelity::default()).collect::<Vec<_>>();
    let mut identical = true;
    for (i, control) in controls.iter_mut().enumerate() {
        let mut placement = allocator.admit(start(i) + lengths[i] + 1)?;
        for part in tokens[..start(i)].chunks(engine.prefill_capacity()) {
            engine.prefill_device(&mut placement, part, transport.as_deref_mut().map(|t| (t, runtime)), 0)?;
        }
        // The ordinary singleton route retains this short unit's final norm in
        // slot 0. Score its rows with the same bounded heads as the packed arm.
        ensure!(lengths[i] < 512, "singleton fidelity control must retain one unit");
        engine.prefill_device(&mut placement, &tokens[start(i)..start(i) + lengths[i]],
            transport.as_deref_mut().map(|t| (t, runtime)), 0)?;
        let digests = score_segment(i, prefill_packing::Segment { first: 0, rows: lengths[i], start: start(i),
            tap_offset: 0, tap_rows: 0 }, control)?;
        let same = packed_digests[i] == digests;
        println!("packed stored golden: member {i} all-row byte digests {}", if same { "identical" } else { "DIFFER" });
        identical &= same;
        allocator.release(placement);
    }
    for (i, (score, control)) in scores.iter().zip(&controls).enumerate() {
        let (kl, top1) = score.metrics()?;
        let (base_kl, base_top1) = control.metrics()?;
        println!("packed stored golden: sequence {i}: {} rows at {}: KL {kl:.8}, top-1 {top1:.4}; singleton KL {base_kl:.8}, top-1 {base_top1:.4}; delta KL {:.8}, delta top-1 {:.4}",
            score.rows, start(i), kl - base_kl, top1 - base_top1);
    }
    let (pooled, control) = (Fidelity::pool(&scores), Fidelity::pool(&controls));
    let (kl, top1) = pooled.metrics()?;
    let (base_kl, base_top1) = control.metrics()?;
    println!("packed stored golden pooled: {} rows, KL {kl:.8}, top-1 {top1:.4}; singleton KL {base_kl:.8}, top-1 {base_top1:.4}; delta KL {:.8}, delta top-1 {:.4}; absolute K4 tier 0.05/0.93 INFORMATIONAL ONLY",
        pooled.rows, kl - base_kl, top1 - base_top1);
    println!("packed stored golden: all-row byte identity {identical}; per-member route drift INFORMATIONAL ONLY");
    pooled.non_inferior(&control).context("packed stored golden pooled")?;
    println!("packed stored golden: pooled singleton non-inferiority PASS (+0.005 KL/-0.01 top-1)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stored_golden_gate_uses_all_member_rows_and_rejects_bad_shapes() -> Result<()> {
        let mut score = Fidelity::default();
        score.add(&[4.0, 0.0, 0.0, 4.0], &[4.0, 0.0, 0.0, 4.0], 2)?;
        assert_eq!((score.rows, score.top1), (2, 2));
        score.non_inferior(&score)?;
        assert!(score.add(&[0.0], &[0.0, 1.0], 2).is_err());
        assert!(score.add(&[f32::NAN, 0.0], &[0.0, 1.0], 2).is_err());
        let mut changed = Fidelity::default();
        changed.add(&[0.0, 4.0], &[4.0, 0.0], 2)?;
        let mut control = Fidelity::default();
        control.add(&[4.0, 0.0], &[4.0, 0.0], 2)?;
        assert!(changed.non_inferior(&control).is_err());
        assert!(Fidelity::default().non_inferior(&Fidelity::default()).is_err());
        assert!(score.non_inferior(&control).is_err());
        Ok(())
    }

    #[test]
    fn singleton_non_inferiority_gates_row_weighted_pool_and_reports_member_drift() -> Result<()> {
        let control = Fidelity { rows: 100, kl_sum: 20.0, top1: 80 };
        let edge = Fidelity { rows: 100, kl_sum: 20.5, top1: 79 };
        edge.non_inferior(&control)?;
        assert!(Fidelity { kl_sum: 20.5001, ..edge }.non_inferior(&control).is_err());
        assert!(Fidelity { top1: 78, ..edge }.non_inferior(&control).is_err());
        assert!(Fidelity { rows: 100, kl_sum: f64::NAN, top1: 80 }.non_inferior(&control).is_err());
        let scores = [Fidelity { rows: 1, kl_sum: 0.02, top1: 0 },
            Fidelity { rows: 99, kl_sum: 0.0, top1: 99 }];
        let controls = [Fidelity { rows: 1, kl_sum: 0.0, top1: 1 },
            Fidelity { rows: 99, kl_sum: 0.0, top1: 99 }];
        // Individual route-switch drift is informational; admission gates the
        // row-weighted pool, not the unweighted mean of member scores.
        assert!(scores[0].non_inferior(&controls[0]).is_err());
        let pooled = Fidelity::pool(&scores);
        assert_eq!((pooled.rows, pooled.top1), (100, 99));
        assert!((pooled.metrics()?.0 - 0.0002).abs() < 1e-12);
        pooled.non_inferior(&Fidelity::pool(&controls))?;
        Ok(())
    }

    #[test]
    fn stored_sequences_wrap_independently() {
        assert_eq!(sequence(&[1, 2, 3], 0, 4), [1, 2, 3, 1]);
        assert_eq!(sequence(&[1, 2, 3], 1, 4), [2, 3, 1, 2]);
    }
}
