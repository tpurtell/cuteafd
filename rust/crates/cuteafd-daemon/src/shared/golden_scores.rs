//! CPU scoring for independently exported, explicitly selected golden rows.

#[cfg(test)]
pub(crate) mod push_regression {
    use anyhow::{ensure, Result};
    use std::path::Path;

    pub(crate) fn prompts(sequences: usize) -> Vec<Vec<u32>> {
        (0..sequences).map(|slot| (0..2067 + slot * 19)
            .map(|i| 1000 + ((i * 37 + slot * 113) % 4096) as u32).collect()).collect()
    }

    pub(crate) fn teacher_rows() -> Vec<u32> { (42..50).collect() }

    fn bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    fn check_logits(values: &[f32], rows: usize, vocab: usize) -> Result<()> {
        ensure!(vocab > 0 && rows.checked_mul(vocab) == Some(values.len())
            && values.iter().all(|value| value.is_finite()),
            "push regression logits are incomplete or nonfinite");
        Ok(())
    }

    pub(crate) fn draft_dump(output: &Path, sequences: usize, width: usize, logit_width: usize, vocab: usize,
        mut propose: impl FnMut() -> Result<(Vec<Vec<u32>>, Vec<f32>)>) -> Result<()> {
        // Some drafters retain an anchor head row as well as their emitted rows.
        ensure!(sequences > 0 && width > 0 && logit_width > 0, "empty push regression draft shape");
        let (tokens, logits) = propose()?;
        ensure!(tokens.len() == sequences && tokens.iter().all(|row| row.len() == width
            && row.iter().all(|&token| (token as usize) < vocab)),
            "push regression draft outputs differ from requested shape");
        check_logits(&logits, sequences * logit_width, vocab)?;
        let (replayed, replay_logits) = propose()?;
        ensure!(tokens == replayed && bytes(&logits) == bytes(&replay_logits),
            "push regression draft replay differs within one version");
        std::fs::create_dir_all(output)?;
        std::fs::write(output.join(format!("drafts-{sequences}.txt")),
            tokens.iter().map(|row| format!("{row:?}\n")).collect::<String>())?;
        std::fs::write(output.join(format!("draft-logits-{sequences}.bin")), bytes(&logits))?;
        Ok(())
    }

    pub(crate) fn target_dump(output: &Path, sequences: usize, vocab: usize, values: &[f32]) -> Result<()> {
        check_logits(values, sequences * teacher_rows().len(), vocab)?;
        std::fs::create_dir_all(output)?;
        std::fs::write(output.join(format!("logits-{sequences}.bin")), bytes(values))?;
        Ok(())
    }

    #[test]
    fn dumps_are_exact_and_reject_changed_replay_and_nonfinite_rows() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let expected = vec![0., 1., 2., 3.];
        draft_dump(dir.path(), 1, 2, 2, 2, || Ok((vec![vec![0, 1]], expected.clone())))?;
        assert_eq!(std::fs::read(dir.path().join("draft-logits-1.bin"))?, bytes(&expected));
        let mut calls = 0;
        assert!(draft_dump(dir.path(), 1, 2, 2, 2, || {
            calls += 1;
            Ok((vec![vec![0, 1]], vec![calls as f32; 4]))
        }).is_err());
        draft_dump(dir.path(), 1, 1, 2, 2, || Ok((vec![vec![1]], expected.clone())))?;
        assert!(draft_dump(dir.path(), 1, 1, 1, 2,
            || Ok((vec![vec![1]], expected.clone()))).is_err());
        assert!(target_dump(dir.path(), 1, 2, &[f32::NAN; 16]).is_err());
        assert!(target_dump(dir.path(), 2, 2, &[0.; 16]).is_err());
        let prompts = prompts(2);
        assert_eq!((prompts[0].len(), prompts[1].len()), (2067, 2086));
        Ok(())
    }
}
use anyhow::{ensure, Result};

pub(crate) fn select_scored_rows(logits: &[f32], positions: &[usize], vocab: usize) -> Result<Vec<f32>> {
    ensure!(vocab > 0 && logits.len() % vocab == 0 && !positions.is_empty(), "dense row selection shape mismatch");
    let rows = logits.len() / vocab;
    ensure!(positions.iter().all(|&p| p > 0 && p <= rows)
        && positions.windows(2).all(|p| p[0] < p[1]), "invalid dense score positions");
    Ok(positions.iter().flat_map(|&p| logits[(p - 1) * vocab..p * vocab].iter().copied()).collect())
}

pub(crate) fn scored_nll(logits: &[f32], targets: &[u32], vocab: usize) -> Result<f64> {
    ensure!(vocab > 0 && !targets.is_empty() && logits.len() == targets.len() * vocab,
        "dense scoring shape mismatch");
    ensure!(logits.iter().all(|v| v.is_finite()) && targets.iter().all(|&t| (t as usize) < vocab),
        "dense scoring requires finite logits and in-range tokens");
    let sum = logits.chunks_exact(vocab).zip(targets).map(|(row, &next)| {
        let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        top + row.iter().map(|&v| (v as f64 - top).exp()).sum::<f64>().ln() - row[next as usize] as f64
    }).sum::<f64>();
    Ok(sum / targets.len() as f64)
}

pub(crate) fn scored_kl(reference: &[f32], logits: &[f32], vocab: usize) -> Result<(f64, f64)> {
    ensure!(vocab > 0 && !reference.is_empty() && reference.len() == logits.len()
        && reference.len() % vocab == 0, "dense KL shape mismatch");
    ensure!(reference.iter().chain(logits).all(|v| v.is_finite()), "dense KL requires finite logits");
    let log_z = |row: &[f32]| {
        let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        top + row.iter().map(|&v| (v as f64 - top).exp()).sum::<f64>().ln()
    };
    let mut sum = 0.0;
    let mut max = 0.0f64;
    for (reference, row) in reference.chunks_exact(vocab).zip(logits.chunks_exact(vocab)) {
        let (reference_z, z) = (log_z(reference), log_z(row));
        let kl = reference.iter().zip(row).map(|(&p, &q)| {
            let log_p = p as f64 - reference_z;
            log_p.exp() * (log_p - (q as f64 - z))
        }).sum::<f64>().max(0.0);
        sum += kl;
        max = max.max(kl);
    }
    Ok((sum / (reference.len() / vocab) as f64, max))
}

