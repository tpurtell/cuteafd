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
use anyhow::{ensure, Context, Result};

/// Full-vocabulary references stay on disk; diagnostics retain at most one row.
pub(crate) struct ReferenceRows {
    file: std::fs::File,
    rows: usize,
    vocab: usize,
}

impl ReferenceRows {
    pub(crate) fn open(path: &std::path::Path, rows: usize, vocab: usize, optional: bool) -> Result<Option<Self>> {
        let file = match std::fs::File::open(path) {
            Ok(file) => file,
            Err(error) if optional && error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).with_context(|| format!("golden logits {}", path.display())),
        };
        let bytes = rows.checked_mul(vocab).and_then(|n| n.checked_mul(4))
            .context("golden logits extent overflow")?;
        ensure!(rows > 0 && vocab > 0 && file.metadata()?.is_file() && file.metadata()?.len() == bytes as u64,
            "golden logits shape differs");
        Ok(Some(Self { file, rows, vocab }))
    }

    pub(crate) fn row(&self, index: usize) -> Result<Vec<f32>> {
        use std::os::unix::fs::FileExt;
        ensure!(index < self.rows, "golden logits row outside reference");
        let mut bytes = vec![0; self.vocab * 4];
        self.file.read_exact_at(&mut bytes, (index * self.vocab * 4) as u64)?;
        let values = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        ensure!(values.iter().all(|v| v.is_finite()), "golden logits are non-finite at row {index}");
        Ok(values)
    }
}

/// Incremental diagnostics preserve the original row-order floating point sums.
#[derive(Default)]
pub(crate) struct RowMetrics {
    pub rows: usize,
    pub scored: usize,
    pub agree: usize,
    pub next_ok: usize,
    pub reference_next: usize,
    pub nll: f64,
    pub reference_nll: f64,
    pub kl: f64,
}

impl RowMetrics {
    pub(crate) fn push(&mut self, ours: &[f32], reference: Option<&[f32]>, next: Option<u32>) -> Result<()> {
        ensure!(!ours.is_empty() && ours.iter().all(|v| v.is_finite()), "engine logits are empty or non-finite");
        let argmax = |values: &[f32]| values.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).unwrap().0;
        let log_z = |values: &[f32]| {
            let top = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            top + values.iter().map(|&v| (v as f64 - top).exp()).sum::<f64>().ln()
        };
        let ours_z = log_z(ours);
        if let Some(next) = next {
            ensure!((next as usize) < ours.len(), "golden next token outside vocabulary");
            self.nll += ours_z - ours[next as usize] as f64;
            self.next_ok += usize::from(argmax(ours) == next as usize);
            self.scored += 1;
        }
        if let Some(reference) = reference {
            ensure!(reference.len() == ours.len() && reference.iter().all(|v| v.is_finite()),
                "golden logits shape/finiteness differs");
            self.agree += usize::from(argmax(ours) == argmax(reference));
            let reference_z = log_z(reference);
            self.kl += reference.iter().zip(ours).map(|(&p, &q)| {
                let p = p as f64 - reference_z;
                p.exp() * (p - (q as f64 - ours_z))
            }).sum::<f64>();
            if let Some(next) = next {
                self.reference_nll += reference_z - reference[next as usize] as f64;
                self.reference_next += usize::from(argmax(reference) == next as usize);
            }
        }
        self.rows += 1;
        Ok(())
    }
}

/// Dense qualification keeps its historical lower-index tie rule and nonnegative row KL.
#[derive(Default)]
pub(crate) struct DenseRowMetrics {
    pub rows: usize,
    pub agree: usize,
    pub nll: f64,
    pub reference_nll: f64,
    pub kl: f64,
    pub max_kl: f64,
}

impl DenseRowMetrics {
    pub(crate) fn push(&mut self, ours: &[f32], reference: &[f32], next: u32) -> Result<()> {
        let nll = scored_nll(ours, &[next], ours.len())?;
        let reference_nll = scored_nll(reference, &[next], ours.len())?;
        let (kl, _) = scored_kl(reference, ours, ours.len())?;
        let argmax = |row: &[f32]| row.iter().enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0))).unwrap().0;
        self.nll += nll;
        self.reference_nll += reference_nll;
        self.kl += kl;
        self.max_kl = self.max_kl.max(kl);
        self.agree += usize::from(argmax(ours) == argmax(reference));
        self.rows += 1;
        Ok(())
    }
}

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

#[cfg(test)]
mod bounded_rows_tests {
    use super::*;

    #[test]
    fn streamed_metrics_preserve_legacy_nll_kl_and_top1_bits() -> Result<()> {
        let (vocab, first) = (5, 1);
        let ours = [2.0, -1.0, 2.0, 0.0, -0.0, -7.0, 3.0, 0.25, 3.0, -4.0];
        let reference = [1.0, 0.0, -2.0, 1.0, 0.0, 2.0, -1.0, 2.0, -0.0, 0.0,
            -6.0, 2.5, 0.25, 2.5, -4.0];
        let tokens = [1, 3, 2];
        let mut metrics = RowMetrics::default();
        for (row, values) in ours.chunks_exact(vocab).enumerate() {
            metrics.push(values, Some(&reference[(first + row) * vocab..][..vocab]),
                tokens.get(first + row + 1).copied())?;
        }
        let argmax = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let (mut agree, mut next_ok, mut reference_next, mut nll, mut scored) = (0, 0, 0, 0.0f64, 0);
        for (row, values) in ours.chunks_exact(vocab).enumerate() {
            let theirs = &reference[(first + row) * vocab..][..vocab];
            agree += usize::from(argmax(values) == argmax(theirs));
            if let Some(&next) = tokens.get(first + row + 1) {
                next_ok += usize::from(argmax(values) == next as usize);
                reference_next += usize::from(argmax(theirs) == next as usize);
                let top = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                nll += top + values.iter().map(|&v| (v as f64 - top).exp()).sum::<f64>().ln()
                    - values[next as usize] as f64;
                scored += 1;
            }
        }
        let legacy = (agree, next_ok, reference_next, nll, scored);
        assert_eq!((metrics.agree, metrics.next_ok, metrics.reference_next, metrics.scored),
            (legacy.0, legacy.1, legacy.2, legacy.4));
        assert_eq!(metrics.nll.to_bits(), legacy.3.to_bits());
        assert_eq!((metrics.kl / metrics.rows as f64).to_bits(),
            crate::families::glm5_flash::mean_kl(&ours, &reference, first, vocab).to_bits());
        Ok(())
    }

    #[test]
    fn streamed_v4_log_softmax_nll_and_digest_keep_original_bits() -> Result<()> {
        use std::hash::{Hash, Hasher};
        let (vocab, rows) = (17, 29);
        let ours = (0..vocab * rows).map(|i| ((i * 37 % 113) as f32 - 56.0) / 7.0).collect::<Vec<_>>();
        let reference = ours.iter().enumerate().map(|(i, &v)| v + (i % 7) as f32 / 13.0).collect::<Vec<_>>();
        let tokens = (0..rows).map(|i| (i * 11 % vocab) as u32).collect::<Vec<_>>();
        let mut metrics = RowMetrics::default();
        let (mut nll, mut reference_nll) = (0.0f64, 0.0f64);
        let (mut streamed_hash, mut legacy_hash) = (std::collections::hash_map::DefaultHasher::new(),
            std::collections::hash_map::DefaultHasher::new());
        let lp = |values: &[f32], token: u32| {
            let top = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let z = top + values.iter().map(|&v| (v as f64 - top).exp()).sum::<f64>().ln();
            values[token as usize] as f64 - z
        };
        for (row, values) in ours.chunks_exact(vocab).enumerate() {
            let theirs = &reference[row * vocab..][..vocab];
            let next = tokens.get(row + 1).copied();
            metrics.push(values, Some(theirs), next)?;
            values.iter().for_each(|v| v.to_bits().hash(&mut streamed_hash));
            if let Some(next) = next {
                nll -= lp(values, next);
                reference_nll -= lp(theirs, next);
            }
        }
        ours.iter().for_each(|v| v.to_bits().hash(&mut legacy_hash));
        assert_eq!(metrics.nll.to_bits(), nll.to_bits());
        assert_eq!(metrics.reference_nll.to_bits(), reference_nll.to_bits());
        assert_eq!(streamed_hash.finish(), legacy_hash.finish());
        assert_eq!(metrics.scored, rows - 1);
        Ok(())
    }

    #[test]
    fn disk_rows_preserve_bits_and_fail_closed_on_bad_extents() -> Result<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("logits.bin");
        assert!(ReferenceRows::open(&path, 2, 2, true)?.is_none());
        assert!(ReferenceRows::open(&path, 2, 2, false).is_err());
        let values = [-0.0f32, 0.0, f32::MIN, f32::MAX];
        std::fs::write(&path, values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
        let rows = ReferenceRows::open(&path, 2, 2, false)?.unwrap();
        for row in 0..2 {
            assert_eq!(rows.row(row)?.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                values[row * 2..][..2].iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        assert!(rows.row(2).is_err());
        assert!(ReferenceRows::open(&path, usize::MAX, 2, false).is_err());
        assert!(ReferenceRows::open(&path, 1, 2, false).is_err());
        std::fs::write(&path, f32::NAN.to_le_bytes())?;
        assert!(ReferenceRows::open(&path, 1, 1, false)?.unwrap().row(0).is_err());
        std::fs::write(&path, [0u8; 3])?;
        assert!(ReferenceRows::open(&path, 1, 1, false).is_err());
        Ok(())
    }

    #[test]
    fn streamed_dense_metrics_preserve_sparse_selection_ties_and_clamped_kl() -> Result<()> {
        let (rows, vocab) = (19, 7);
        let ours = (0..rows * vocab).map(|i| ((i * 37 % 113) as f32 - 56.0) / 7.0).collect::<Vec<_>>();
        let reference = ours.iter().enumerate().map(|(i, &v)| v + (i % 7) as f32 / 13.0).collect::<Vec<_>>();
        let positions = [1, 3, 11, 18];
        let targets = [2, 5, 0, 6];
        let selected = select_scored_rows(&ours, &positions, vocab)?;
        let theirs = select_scored_rows(&reference, &positions, vocab)?;
        let mut metrics = DenseRowMetrics::default();
        for ((values, reference), &target) in selected.chunks_exact(vocab).zip(theirs.chunks_exact(vocab)).zip(&targets) {
            metrics.push(values, reference, target)?;
        }
        assert_eq!((metrics.nll / metrics.rows as f64).to_bits(), scored_nll(&selected, &targets, vocab)?.to_bits());
        assert_eq!((metrics.reference_nll / metrics.rows as f64).to_bits(), scored_nll(&theirs, &targets, vocab)?.to_bits());
        let (mean, max) = scored_kl(&theirs, &selected, vocab)?;
        assert_eq!((metrics.kl / metrics.rows as f64).to_bits(), mean.to_bits());
        assert_eq!(metrics.max_kl.to_bits(), max.to_bits());
        let mut tie = DenseRowMetrics::default();
        tie.push(&[1.0, 1.0], &[1.0, 0.0], 0)?;
        assert_eq!(tie.agree, 1);
        assert!(tie.push(&[f32::NAN], &[0.0], 0).is_err());
        assert!(tie.push(&[0.0], &[0.0, 0.0], 0).is_err());
        Ok(())
    }

    #[test]
    fn metrics_reject_nonfinite_mismatched_and_out_of_vocab_rows() {
        for (ours, reference, next) in [(&[][..], None, None),
            (&[f32::NAN][..], None, None), (&[0.0][..], Some(&[0.0, 1.0][..]), None),
            (&[0.0][..], Some(&[f32::INFINITY][..]), None), (&[0.0][..], None, Some(1))] {
            assert!(RowMetrics::default().push(ours, reference, next).is_err());
        }
    }
}
