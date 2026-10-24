//! Serve-loop parts a family scheduler composes: independent decode lanes
//! ([`lanes`]: join without cancellation, cohort rebalance, the two-part lane
//! commit) and their shared owners ([`decode_lanes`]: request bank and round
//! guard, aggregate failure boundary, phase-major view retirement, lane
//! assignment).
pub(crate) mod decode_lanes;
pub(crate) mod lanes;
