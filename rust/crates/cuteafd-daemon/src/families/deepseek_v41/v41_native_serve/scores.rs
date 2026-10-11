//! V4.1's vocabulary width for the shared `ScoreRows`/`RetainedScores`.
pub(crate) use crate::shared::token_io::scores::{RetainedScores, ScoreRows};

pub(crate) const VOCAB: usize = 129_280;
pub(crate) const ROW_BYTES: usize = VOCAB * 4;
