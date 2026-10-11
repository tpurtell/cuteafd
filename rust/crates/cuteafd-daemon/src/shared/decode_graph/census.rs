//! Opt-in diagnostics; the live control file is read only at idle boundaries.
use std::sync::{atomic::{AtomicU8, Ordering}, OnceLock};

static ENABLED: OnceLock<bool> = OnceLock::new();
static EAGER: AtomicU8 = AtomicU8::new(0);

pub(crate) fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var_os("CUTEAFD_GRAPH_CENSUS").is_some_and(|v| v == "1"))
}

fn bank_id(bank: &str) -> Option<u8> {
    Some(match bank { "baseline" => 0, "sparse" => 1, "index" => 2,
        "layer_row" => 3, "head" => 4, "dspark" => 5, "window" => 6, _ => return None })
}

pub(crate) fn idle() {
    if !enabled() { return }
    let Some(path) = std::env::var_os("CUTEAFD_GRAPH_CENSUS_CONTROL") else { return };
    match std::fs::read_to_string(path).ok().and_then(|s| bank_id(s.trim())) {
        Some(bank) => {
            if EAGER.swap(bank, Ordering::Relaxed) != bank {
                tracing::info!(target: "cuteafd::graph_capture", bank, "graph census idle arm switch");
            }
        }
        None => tracing::error!(target: "cuteafd::graph_capture", "invalid graph census control; retaining arm"),
    }
}

pub(crate) fn eager(bank: &str) -> bool {
    enabled() && bank_id(bank).is_some_and(|id| id != 0 && id == EAGER.load(Ordering::Relaxed))
}

pub(crate) fn arm() -> u8 { EAGER.load(Ordering::Relaxed) }

/// # Safety
/// Inputs and captured storage remain live through queued work on this stream.
pub(crate) unsafe fn dispatch(library: &cuteafd_ffi::NativeLibrary,
    stream: *mut std::ffi::c_void, bank: &std::ffi::CStr, rows: usize,
    graph: Option<*mut std::ffi::c_void>, eager_call: impl FnOnce() -> anyhow::Result<()>) -> anyhow::Result<()> {
    let replay = graph.is_some() && !eager(bank.to_str().expect("static census bank"));
    let call = || {
        if replay {
            // SAFETY: caller pins the executable and stream through queued use.
            unsafe { library.cuda_graph_launch(graph.unwrap(), stream) }
        } else { eager_call() }
    };
    if enabled() {
        // SAFETY: caller owns this stream and the closure preserves its device.
        unsafe { library.cuda_graph_census_call(stream, bank, rows, replay, arm(), call) }
    } else { call() }
}

#[cfg(test)]
mod tests {
    #[test]
    fn unknown_control_does_not_select_a_bank() {
        assert_eq!(super::bank_id("baseline"), Some(0));
        for name in ["sparse", "index", "layer_row", "head", "dspark", "window"] {
            assert!(super::bank_id(name).is_some_and(|id| id > 0));
        }
        assert_eq!(super::bank_id("all"), None);
        assert_eq!(super::bank_id("../sparse"), None);
    }
}
