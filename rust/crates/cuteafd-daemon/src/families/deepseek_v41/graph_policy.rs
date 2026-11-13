//! Captured shapes per backbone layer, sharing the containing lane's buffers.
use anyhow::{ensure, Result};

pub(super) const MAX_DECODE_ROWS: u32 = cuteafd_core::V41_VERIFY_ROWS as u32;

/// Rows of one full prefill chunk. Above the decode bank a live request captures
/// only this and the 128-row decoder replay, the two large shapes every long
/// prompt repeats. Unset (tests, tools), every shape keeps capturing.
static PREFILL_ROWS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();

/// Startup: register the serving prefill chunk. A prompt's tail chunk has an
/// odd row count that no later request repeats, so it executes eagerly instead
/// of instantiating one single-use graph per layer and site (#410 A).
pub(super) fn set_prefill_rows(rows: u32) -> Result<()> {
    ensure!(rows > 0, "prefill chunk rows must be positive");
    match PREFILL_ROWS.get() {
        Some(&set) => ensure!(set == rows, "prefill chunk rows already set to {set}"),
        None => { let _ = PREFILL_ROWS.set(rows); }
    }
    Ok(())
}

thread_local! {
    // Startup owns a dedicated CUDA worker thread. Avoid threading a layout
    // field through every primitive owner; attach refreshes it before capture.
    static RANKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
pub(crate) fn ranks() -> usize { RANKS.with(std::cell::Cell::get) }
fn set_layout(ranks: usize) { RANKS.with(|layout| layout.set(ranks)); }

pub(crate) fn pinned_shape(device: i32, rows: u32) -> bool {
    let Ok(rank) = usize::try_from(device) else { return false };
    let Some(list) = cuteafd_loader::serving_capacity::v41_graphs::v41_prewarm(ranks(), rank) else { return false };
    list.entries.iter().chain(&list.prefill_entries)
        .any(|entry| entry.rows == rows as usize && entry.pin)
}

pub(super) fn captures_shape(rows: u32) -> bool {
    large_shape_captured(rows, PREFILL_ROWS.get().copied())
}

/// Only slot zero serves non-unit prompt, replay and continuation work.
pub(crate) const PREFILL_PREWARM_SLOTS: [usize; 1] = [0];
/// The terminal projects the selected last row, not the whole backbone chunk.
pub(crate) const PREFILL_HEAD_PREWARM_ROWS: [usize; 1] = [1];

/// Nonwindowed prompt/replay paths reach these exact row counts. Streamed
/// encoder units have offset bindings and deliberately remain eager (#1638).
pub(crate) fn prefill_prewarm_rows(rank: usize, capacity: usize) -> Vec<usize> {
    prefill_rows(rank, capacity, PREFILL_ROWS.get().copied())
}

fn prefill_rows(rank: usize, capacity: usize, chunk: Option<u32>) -> Vec<usize> {
    cuteafd_loader::serving_capacity::v41_graphs::v41_prefill_rows(ranks(), rank, capacity, chunk.map(|row| row as usize))
}

/// Records one eager run of a shape the policy would graph but this owner's
/// bank cannot capture. Every site that chooses eager for that reason reports
/// through here, so `/v1/stats["graph_missing_variants"]` holds one entry per
/// variant and the first execution logs one structured warning.
pub(super) fn note_miss(shape: String, context: impl FnOnce() -> String) {
    crate::shared::decode_graph::note_eager("deepseek_v41", &shape, context);
}

fn large_shape_captured(rows: u32, prefill: Option<u32>) -> bool {
    rows <= MAX_DECODE_ROWS || rows == cuteafd_core::coordinator_programs::V41_DECODER_REPLAY_ROWS
        || prefill.is_none_or(|chunk| rows == chunk)
}

fn site_name(name: &'static str) -> &'static str {
    match name {
        "window" => "deepseek_v41.window", "compressor" => "deepseek_v41.compressor",
        "index_selection" => "deepseek_v41.index_selection",
        "target_head" => "deepseek_v41.target_head", "target_norm" => "deepseek_v41.target_norm",
        "target_embedding" => "deepseek_v41.target_embedding", "tp2_ffn" => "deepseek_v41.tp2_ffn",
        "vocabulary_logits" => "deepseek_v41.vocabulary_logits", "vocabulary_greedy" => "deepseek_v41.vocabulary_greedy",
        "sparse_attention" => "deepseek_v41.sparse_attention", "dual_attention_tail" => "deepseek_v41.dual_attention_tail",
        "v41.attention.query" => "deepseek_v41.attention.query",
        "v41.attention.query.tp2_prefix" => "deepseek_v41.attention.query.tp2_prefix",
        "v41.attention.output" => "deepseek_v41.attention.output",
        "v41.backbone.shared" => "deepseek_v41.backbone.shared",
        "v41.router.remote" => "deepseek_v41.router.remote",
        "v41.router.local" => "deepseek_v41.router.local",
        "v41.index.query" => "deepseek_v41.index.query", "v41.engram.gate" => "deepseek_v41.engram.gate",
        "v41.projection.query.rank" => "deepseek_v41.projection.query.rank",
        "v41.projection.query.gather" => "deepseek_v41.projection.query.gather",
        "v41.projection.output.rank" => "deepseek_v41.projection.output.rank",
        "v41.projection.output.gather" => "deepseek_v41.projection.output.gather",
        "dspark_chain" => "deepseek_v41.dspark_chain",
        "dspark_cache" => "deepseek_v41.dspark_cache",
        "dspark_distributed_terminal" => "deepseek_v41.dspark_distributed_terminal",
        // Empty standalone banks are visited too; they have no serving capture sites.
        _ => name,
    }
}

pub(crate) fn attach(library: &cuteafd_ffi::NativeLibrary, ranks: usize,
    mut visit: impl FnMut(&mut crate::shared::decode_graph::device::GraphVisitor<'_>) -> Result<()>) -> Result<()> {
    use crate::shared::decode_graph::device::{Basis, Cycle, Preference, Site};
    tracing::warn!("V4.1 graph cache uses byte estimates: SM120 node rates are not calibrated");
    set_layout(ranks);
    visit(&mut |device, name, spans, bank| {
        let (base, extra) = cuteafd_loader::placement::graph_reserve::v41_default(ranks, device as usize);
        let policy = cuteafd_loader::placement::graph_reserve::current(device as usize,
            false, base, extra).map_err(anyhow::Error::msg)?;
        bank.attach(library, device, policy.bytes, Site { name: site_name(name), cycle: Cycle::Other,
            preference: Preference::Adaptive, bootstrap: 2 << 20, basis: Basis::Bytes,
            spans_devices: spans || ranks > 1 })?;
        crate::shared::decode_graph::device::set_reserve_policy(device, policy)
    })
}

pub(crate) fn cycle(cycle: crate::shared::decode_graph::device::Cycle,
    mut visit: impl FnMut(&mut crate::shared::decode_graph::device::GraphVisitor<'_>) -> Result<()>) -> Result<()> {
    visit(&mut |_, _, _, bank| { bank.cycle(cycle); Ok(()) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_state_is_local_to_the_engine_thread_and_refreshes_on_restart() {
        std::thread::spawn(|| {
            assert_eq!(ranks(), 0);
            set_layout(1);
            assert_eq!(ranks(), 1);
            std::thread::spawn(|| assert_eq!(ranks(), 0)).join().unwrap();
            set_layout(2);
            assert_eq!(ranks(), 2);
        }).join().unwrap();
    }

    #[test]
    fn compressor_chain_counts_only_the_successful_replay_lookup() {
        let source = include_str!("v41_compressor.rs");
        let start = source.find("if crate::shared::memory::chain::active() && self.pending_query.is_none()").unwrap();
        let end = start + source[start..].find("crate::shared::memory::chain::settle").unwrap();
        let chain = &source[start..end];
        // Liveness validation must not count a hit or keep an evicted pointer.
        assert_eq!(chain.matches("self.selected_graph()").count(), 1);
        assert_eq!(chain.matches("self.retained_graphs.cached_launch(&(prepared.rows, state.owner))").count(), 1);
        assert!(chain.find("cached_launch").unwrap() < chain.find("census::dispatch").unwrap());
        let select = &source[source.find("fn select_graph(").unwrap()..source.find("fn selected_graph(").unwrap()];
        assert_eq!(select.matches("cached_launch").count(), 1);
        let selected = &source[source.find("fn selected_graph(").unwrap()..];
        let selected = &selected[..selected.find("\n    }").unwrap()];
        assert!(!selected.contains("cached_launch"));
    }

    #[test]
    fn startup_prefill_keys_are_reachable_from_slot_zero_and_selected_head_row() {
        use crate::shared::memory::row_window::RowWindow;
        // Enumerate the geometry consumed by startup, including a chunk that
        // coincides with replay and a capacity that excludes the chunk.
        for (capacity, chunk, expected) in [(2048, 2048, vec![128]),
            (128, 128, vec![128]), (128, 2048, vec![128])] {
            let rows = cuteafd_loader::serving_capacity::v41_graphs::v41_prefill_rows(1, 0, capacity, Some(chunk as usize));
            assert_eq!(rows, expected);
            let keys: Vec<_> = PREFILL_PREWARM_SLOTS.into_iter().flat_map(|slot| {
                rows.iter().map(move |&rows| (slot, rows, RowWindow::NONE, false))
            }).collect();
            assert!(!keys.is_empty());
            for (slot, rows, window, attention_decode) in keys {
                assert_eq!(slot, 0);
                assert!(!window.is_windowed());
                assert!(!attention_decode);
                assert!(rows == 128 || rows == chunk as usize);
            }
            let head_keys: Vec<_> = PREFILL_PREWARM_SLOTS.into_iter()
                .flat_map(|slot| PREFILL_HEAD_PREWARM_ROWS.map(|rows| (slot, rows))).collect();
            assert_eq!(head_keys, vec![(0, 1)]);
        }
        // Tie the enumeration to both live startup implementations rather than
        // letting a correct policy test hide an unreachable caller inventory.
        for source in [include_str!("v41_native_serve.rs"), include_str!("v41_native_serve/distributed.rs")] {
            let at = source.find("for index in ").unwrap();
            let end = source[at..].find("if let Some(draft)").unwrap() + at;
            let startup = &source[at..end];
            assert!(startup.contains("graph_policy::PREFILL_PREWARM_SLOTS"));
            assert!(startup.contains("&mut prefill_slots.slots[index]"));
            assert_eq!(source.matches("slot.capture_startup_prefill(").count(), 1);
        }
        for source in [include_str!("v41_target_pass.rs"), include_str!("v41_target_pass/distributed.rs")] {
            let at = source.find("pub(crate) fn capture_startup_prefill").unwrap();
            let end = source[at..].find("pub(crate) fn capture_startup(").unwrap() + at;
            let startup = &source[at..end];
            assert!(startup.contains("select_window(crate::shared::memory::row_window::RowWindow::NONE)?"));
            if source.contains("resources.head.capture_startup(") {
                assert!(startup.contains("graph_policy::PREFILL_HEAD_PREWARM_ROWS"));
            } else {
                assert!(!startup.contains("head.capture_startup("));
            }
            assert!(!startup.contains("head.capture_startup(&rows)"));
        }
        let lane = include_str!("v41_backbone_lane.rs");
        let at = lane.find("pub(crate) fn capture_startup_prefill").unwrap();
        let end = lane[at..].find("pub(crate) fn startup_prefill_rows").unwrap() + at;
        assert!(lane[at..end].contains("capture_startup_kind(&[], &[], false, false, local)"));
    }

    #[test]
    fn prefill_tail_chunks_execute_eagerly_while_decode_and_full_chunks_capture() {
        for rows in [1, 6, 38, MAX_DECODE_ROWS] {
            assert!(large_shape_captured(rows, Some(2048)));
        }
        assert!(large_shape_captured(2048, Some(2048)));
        assert!(large_shape_captured(128, Some(2048)));
        for rows in [MAX_DECODE_ROWS + 1, 1982, 2047] {
            assert!(!large_shape_captured(rows, Some(2048)));
        }
        assert!(large_shape_captured(1982, None));
    }

}
