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
use anyhow::{bail, ensure, Context, Result};

/// Refuses a reference logits row that carries no information: a non-finite
/// value, or every logit equal (a uniform softmax, which is what an all-zero
/// placeholder file scores as; #2719). A KL against such a row is a number
/// about the engine alone, so the golden must fail instead of printing it.
pub(crate) fn check_reference_row(path: &std::path::Path, row: usize, values: &[f32]) -> Result<()> {
    if let Some(column) = values.iter().position(|v| !v.is_finite()) {
        bail!("golden reference {}: row {row} column {column} is non-finite ({}); \
            regenerate it from the family's reference golden.py", path.display(), values[column]);
    }
    let first = *values.first().with_context(|| format!("golden reference {}: row {row} is empty", path.display()))?;
    if values.iter().all(|&v| v == first) {
        let what = if row == 0 && first == 0.0 { "is all zeros from row 0 (a shape-only placeholder?)" }
            else { "is constant" };
        bail!("golden reference {}: row {row} {what}: every one of its {} logits is {first}, so KL, top-1 and \
            NLL against it carry no information; regenerate it from the family's reference golden.py (#2719)",
            path.display(), values.len());
    }
    Ok(())
}

/// Checks every row of an in-memory [rows, vocab] F32 reference.
pub(crate) fn check_reference_logits(path: &std::path::Path, values: &[f32], vocab: usize) -> Result<()> {
    ensure!(vocab > 0 && !values.is_empty() && values.len().is_multiple_of(vocab),
        "golden reference {}: {} values are not whole rows of {vocab}", path.display(), values.len());
    values.chunks_exact(vocab).enumerate().try_for_each(|(row, values)| check_reference_row(path, row, values))
}

/// Reads and checks a whole [rows, vocab] F32 reference (`logits.bin`).
pub(crate) fn read_reference_logits(path: &std::path::Path, vocab: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("golden logits {}", path.display()))?;
    ensure!(bytes.len().is_multiple_of(4), "golden reference {}: not whole F32 values", path.display());
    let values = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    check_reference_logits(path, &values, vocab)?;
    Ok(values)
}

/// Full-vocabulary references stay on disk; diagnostics retain at most one row.
pub(crate) struct ReferenceRows {
    file: std::fs::File,
    rows: usize,
    vocab: usize,
}

impl ReferenceRows {
    /// Opens and checks every row once (one row resident at a time), so a
    /// placeholder or degenerate reference fails before any engine work.
    pub(crate) fn open(path: &std::path::Path, rows: usize, vocab: usize, optional: bool) -> Result<Option<Self>> {
        use std::io::Read;
        let file = match std::fs::File::open(path) {
            Ok(file) => file,
            Err(error) if optional && error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).with_context(|| format!("golden logits {}", path.display())),
        };
        let bytes = rows.checked_mul(vocab).and_then(|n| n.checked_mul(4))
            .context("golden logits extent overflow")?;
        ensure!(rows > 0 && vocab > 0 && file.metadata()?.is_file() && file.metadata()?.len() == bytes as u64,
            "golden logits shape differs");
        let mut reader = std::io::BufReader::with_capacity(1 << 24, &file);
        let mut raw = vec![0u8; vocab * 4];
        for row in 0..rows {
            reader.read_exact(&mut raw)?;
            let values = raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
            check_reference_row(path, row, &values)?;
        }
        drop(reader);
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

/// Printed once after any legacy-fixture score line.
pub(crate) const CONTEXT_ONLY_NOTE: &str =
    "context-only fixture: no quality verdict (use the published fidelity set)";

impl RowMetrics {
    /// The score line for a legacy `tokens.bin` + `logits.bin` fixture, which has no
    /// generated-row roles: every row is prompt context. Teacher-forced NLL of the
    /// actual next tokens (engine vs reference, same tokens) is a sanity bound; the
    /// reference KL and top-1 are deliberately not reported, since a headline from
    /// context rows reads as a quality score (the verdict gates generated rows only).
    pub(crate) fn context_only_line(&self, label: &str) -> String {
        let scored = self.scored.max(1) as f64;
        let (engine, reference) = (self.nll / scored, self.reference_nll / scored);
        format!("{label} logits (context-only, {} rows, {} scored): NLL sanity bound engine {engine:.4} \
            reference {reference:.4} delta {:+.4} (not a verdict)\n{CONTEXT_ONLY_NOTE}",
            self.rows, self.scored, engine - reference)
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
    fn context_only_line_reports_nll_bound_without_kl_or_top1() -> Result<()> {
        let mut metrics = RowMetrics::default();
        metrics.push(&[2.0, 0.0, -1.0], Some(&[1.0, 0.5, -1.0]), Some(0))?;
        metrics.push(&[0.0, 1.0, 3.0], Some(&[0.0, 2.0, 1.0]), Some(2))?;
        let line = metrics.context_only_line("prefill");
        assert!(line.contains("2 rows, 2 scored") && line.contains("NLL sanity bound engine"), "{line}");
        assert!(line.contains("delta ") && line.contains("not a verdict"), "{line}");
        assert!(line.ends_with("context-only fixture: no quality verdict (use the published fidelity set)"), "{line}");
        let lower = line.to_lowercase();
        assert!(!lower.contains("kl") && !lower.contains("top-1") && !lower.contains("agreement"), "{line}");
        Ok(())
    }

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
        let values = [-0.0f32, f32::MIN_POSITIVE, f32::MIN, f32::MAX];
        std::fs::write(&path, values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
        let rows = ReferenceRows::open(&path, 2, 2, false)?.unwrap();
        for row in 0..2 {
            assert_eq!(rows.row(row)?.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                values[row * 2..][..2].iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        assert!(rows.row(2).is_err());
        assert!(ReferenceRows::open(&path, usize::MAX, 2, false).is_err());
        assert!(ReferenceRows::open(&path, 1, 2, false).is_err());
        std::fs::write(&path, [f32::NAN, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
        assert!(ReferenceRows::open(&path, 1, 2, false).is_err());
        std::fs::write(&path, [0u8; 3])?;
        assert!(ReferenceRows::open(&path, 1, 1, false).is_err());
        Ok(())
    }

    /// #2719: a shape-only placeholder (all zeros), a constant row or a
    /// non-finite value fails at open, naming the file, before any scoring.
    #[test]
    fn placeholder_constant_and_nonfinite_references_fail_at_open() -> Result<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("logits.bin");
        let write = |values: &[f32]| std::fs::write(&path, values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let message = |rows, vocab| ReferenceRows::open(&path, rows, vocab, true).err().map(|e| format!("{e:#}"));
        write(&[0.0; 12])?;
        let zero = message(3, 4).expect("all-zero reference must fail");
        assert!(zero.contains(&path.display().to_string()) && zero.contains("row 0") && zero.contains("all zeros"), "{zero}");
        assert!(read_reference_logits(&path, 4).is_err());
        write(&[1.0, 2.0, 3.0, 4.0, 7.5, 7.5, 7.5, 7.5, 1.0, 0.0, 0.0, 0.0])?;
        let constant = message(3, 4).expect("a constant row must fail");
        assert!(constant.contains("row 1 is constant") && constant.contains("7.5"), "{constant}");
        assert!(check_reference_logits(&path, &[1.0, 2.0, 3.0, 3.0], 2).unwrap_err().to_string().contains("row 1"));
        write(&[1.0, 2.0, 3.0, 4.0, 1.0, f32::INFINITY, 0.0, 0.0])?;
        let infinite = message(2, 4).expect("a non-finite value must fail");
        assert!(infinite.contains("row 1 column 1 is non-finite"), "{infinite}");
        // One distinct logit anywhere in a row makes it informative.
        write(&[0.0, 0.0, 0.0, -1e-30, 5.0, 4.0, 3.0, 2.0])?;
        assert!(ReferenceRows::open(&path, 2, 4, false)?.is_some());
        assert_eq!(read_reference_logits(&path, 4)?.len(), 8);
        assert!(check_reference_logits(&path, &[1.0, 2.0, 3.0], 2).is_err());
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
