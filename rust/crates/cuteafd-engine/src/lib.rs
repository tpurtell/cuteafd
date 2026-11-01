//! Model-agnostic serve runtime of the generic engine (PLAN.md "Execution engines"). Today it
//! holds the prefix cache every generic family shares ([`prefix`]); V4.1 keeps its specialized
//! cache in `cuteafd-daemon::v41_native_serve::prefix`.
pub mod prefix;
pub mod media;
pub mod reserve;
