//! V4.1's retained scores: the shared `ScoreRows`/`RetainedScores` at the
//! family's fixed vocabulary. The old names stay as aliases for one step.
pub(crate) use crate::shared::token_io::scores::{RetainedScores as TokenScores, ScoreRows as BatchScores};

pub(crate) const VOCAB: usize = 129_280;
pub(crate) const ROW_BYTES: usize = VOCAB * 4;
