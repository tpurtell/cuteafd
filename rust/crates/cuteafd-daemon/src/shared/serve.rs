//! Serve-loop parts a family scheduler composes: independent decode lanes
//! ([`lanes`]) and the two-part lane commit ([`lanes::LaneCommit`]).
pub(crate) mod lanes;
