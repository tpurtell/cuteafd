//! Bounded asynchronous usage storage; SQLite never runs on the serving thread.
pub mod store;
pub use store::{Clock, Error, Settings, Store, SystemClock};
