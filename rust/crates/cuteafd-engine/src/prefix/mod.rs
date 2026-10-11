//! The prefix cache every generic family shares (PLAN.md "Prefix cache for every family").
//!
//! Keys are raw token ids (`cuteafd_core::prefix::Retention` with a per-family
//! `ReuseRule`). A snapshot is `len` rows of refcounted device pages ([`RefPagePool`]) plus a
//! copied positional "mark" ([`Mark`]: in a device arena, [`MarkArena`], or in pages of the same
//! pool, [`MarkStore::Pool`]) and what follows it ([`After`]).
//! Families plug in through [`PrefixFamily`]; [`PrefixCache`] does lookup, fork, restore,
//! capture, eviction and the pinned host tier (`cuteafd-hostcache`, generalized by
//! [`FamilyLayout::host_layout`], pages identified by a 64-token hash chain).
mod admission;
mod cache;
mod chain;
mod entry;
mod family;
mod marks;
mod pages;
mod points;
#[cfg(test)]
mod tests;

pub use admission::{fit_output, AdmissionPoll, AdmissionStats, DeferredAdmission};
pub use cache::{Admitted, Captured, Hit, HostPayload, PrefixCache, PrefixConfig, PrefixError, PrefixStats, Source};
pub use chain::{content_id, page_chain, CONTENT_CLASS};
pub use cuteafd_core::prefix::{ReuseRule, SnapshotKind};
pub use entry::{greedy, victim, After, EntryId, Mark};
pub use family::{BoxError, CaptureTicket, Eviction, FamilyLayout, LaggedState, MarkStore, PrefixFamily, RestoreCandidate,
    RestoreContext, RestoreFidelity, RestorePlan};
pub use marks::{ArenaExhausted, MarkArena, MarkSlot};
pub use pages::{Fork, FreedPage, PoolExhausted, RefPagePool, TailCopy};
pub use points::{message_boundaries, plan as plan_points, plan_media as plan_media_points, PointPlan, PointPolicy};
