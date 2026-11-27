//! Family-local qualification on the serving backend and shared genuine-row gate.
use super::{backend::{DraftState, MuseBackend}, prefix::Placement, serve::ServeArgs};
use crate::shared::{prefill_packing::PackingMode, qwen::{packed_check, scheduler::{Backend, PrefillSeq}}};
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::{MarkSlot, PrefixFamily, RefPagePool};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use sha2::{Digest, Sha256};

const CONTINUATION_ROWS: usize = 16;

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)] pub engine: ServeArgs,
    /// Immutable Muse fidelity publication directory.
    #[arg(long)] pub golden: std::path::PathBuf,
    #[arg(long, default_value_t = 4)] pub packed_check: usize,
    #[arg(long)] pub window: Option<String>,
}

pub(crate) async fn run(args: GoldenArgs) -> Result<()> {
    tokio::task::spawn_blocking(move || qualify(args)).await?
}

fn qualify(mut args: GoldenArgs) -> Result<()> {
    use cuteafd_bench::fidelity_dataset as dataset;
    let members = args.packed_check;
    ensure!((2..=8).contains(&members), "Muse --packed-check requires 2..=8 members");
    args.engine.max_sequences = args.engine.max_sequences.max(u8::try_from(2 * members + 1)?);
    args.engine.prefix.resolve_entries(usize::from(args.engine.max_sequences));
    args.engine.validate()?;
    let bytes = std::fs::read(args.golden.join("manifest.json"))?;
    ensure!(format!("{:x}", Sha256::digest(&bytes)) == dataset::MUSE_SPARK_MANIFEST_SHA256,
        "Muse fidelity manifest differs from the immutable publication");
    let manifest: serde_json::Value = serde_json::from_slice(&bytes)?;
    ensure!(manifest["config"] == dataset::MUSE_SPARK_CONFIG, "Muse fidelity config differs");
    let reference = dataset::load(&args.golden, &manifest)?;
    let windows = &reference.windows;
    let first = args.window.as_deref().unwrap_or(&windows.first().context("empty Muse reference")?.id);
    let selected = packed_check::reference_windows(windows, first, members,
        args.engine.prefill_packing == PackingMode::Unrestricted)?;
    ensure!(selected.iter().all(|w| w.media.is_empty()), "Muse packing gate requires original text windows");
    super::serve::with_backend(&args.engine, |backend, slots, _| {
        clear_rings(backend)?;
        ensure!(slots > 2 * members, "Muse packing check needs a spare restore slot");
        let (family, mut cache) = backend.prefix_cache(&args.engine.prefix, slots)?;
        let prompts = packed_check::prompts(&selected[0].tokens, members)?;
        let continuation_capacity = CONTINUATION_ROWS.max(backend.diagnostic_graph_width().unwrap_or(0) + 1);
        let mut allocate = |lengths: Vec<usize>, first_slot: usize| -> Result<Vec<Placement>> {
            lengths.into_iter().enumerate().map(|(i, len)| Ok(cache.admit_cold(&family, len, len + continuation_capacity + 1,
                |pages| Placement { pages, slot: first_slot + i, ..Default::default() })?.placement)).collect()
        };
        let packed = allocate(prompts.iter().map(Vec::len).collect(), 0)?;
        let alone = allocate(prompts.iter().map(Vec::len).collect(), members)?;
        let limits = backend.checked_packed_limits()?.context("Muse packed limits")?;
        let neural = args.engine.draft.is_some();
        let continuation = selected[0].tokens.iter().copied().take(CONTINUATION_ROWS).collect::<Vec<_>>();
        ensure!(continuation.len() == CONTINUATION_ROWS, "Muse cache check needs 16 continuation tokens");
        ensure!(family.slots() >= 2, "Muse packing restore gate requires two admitted prefix mark slots");
        packed_check::check::<_, DraftState>(backend.library(), &prompts, packed, alone, limits, neural,
            args.engine.prefill_packing == PackingMode::Exact,
            |s| singleton(backend, s, cache.pool()),
            |s, callback| packed_unit(backend, s, cache.pool(), callback),
            |p| snapshot_and_restore(backend, &family, p, 2 * members, cache.pool(), &continuation))?;
        if args.engine.prefill_packing == PackingMode::Exact {
            println!("exact Muse packing fails closed on EXL3 at this pin: {}", backend.packed_fallback_reason);
            return Ok(());
        }
        // A fresh pool owns fresh physical pages; no stale synthetic prefix becomes a control.
        drop(cache);
        drop(family);
        clear_rings(backend)?;
        let (family, mut cache) = backend.prefix_cache(&args.engine.prefix, slots)?;
        let mut allocate = |first_slot: usize| -> Result<Vec<Placement>> {
            selected.iter().enumerate().map(|(i, w)| Ok(cache.admit_cold(&family, w.tokens.len(), w.tokens.len(),
                |pages| Placement { pages, slot: first_slot + i, ..Default::default() })?.placement)).collect()
        };
        let placements = allocate(0)?;
        let controls = allocate(members)?;
        let precision = cuteafd_bench::fidelity_precision::resolve("muse-packed-check", Some(&args.engine.snapshot),
            &serde_json::json!([]), "meta-models/Muse-Spark-Open-816B-A42B", None)?;
        packed_check::reference::<_, DraftState>(backend.library(), &selected, placements, controls, limits, neural, &precision,
            |s| singleton(backend, s, cache.pool()),
            |s, callback| packed_unit(backend, s, cache.pool(), callback))?;
        let wide = backend.geometry().decode_rows;
        if wide > cuteafd_loader::families::muse_spark::lane::HEAD_ROWS {
            drop(cache);
            drop(family);
            let windows: Vec<&[u32]> = selected.iter().map(|w| w.tokens.as_slice()).collect();
            check_wide_decode_rows(backend, &args.engine.prefix, slots, &windows, wide)?;
        }
        Ok(())
    })
}

/// Partition `total` verify rows across sequences of at most `SWA_RING_STEP` rows each.
fn wide_verify_partition(total: usize) -> Vec<usize> {
    let per_seq_cap = 32usize.min(cuteafd_loader::families::muse_spark::cache::SWA_RING_STEP as usize);
    let seqs = total.div_ceil(per_seq_cap).max(2);
    let base = total / seqs;
    let extra = total % seqs;
    (0..seqs).map(|i| base + usize::from(i < extra)).collect()
}

/// Qualify wide verify batches (`decode_rows > 64`): verify capture/replay/eager
/// bit-identity at the full admitted `decode_rows` width and compare each
/// sequence's rows against its single-sequence eager run.
fn check_wide_decode_rows(
    backend: &MuseBackend<'_, '_>,
    prefix_args: &crate::shared::prefix::PrefixArgs,
    slots: usize,
    windows: &[&[u32]],
    wide: usize,
) -> Result<()> {
    let counts = wide_verify_partition(wide);
    ensure!(slots >= counts.len(), "Muse wide decode check needs {} slots (admitted {slots})", counts.len());
    let prompt_base = 64usize;
    let mut seq_inputs = Vec::with_capacity(counts.len());
    for (i, &rows) in counts.iter().enumerate() {
        let tokens = windows[i % windows.len()];
        let prompt_len = prompt_base + i;
        ensure!(tokens.len() >= prompt_len + rows,
            "Muse reference window {} too short for wide decode check ({} < {})", i, tokens.len(), prompt_len + rows);
        seq_inputs.push((&tokens[..prompt_len], &tokens[prompt_len..prompt_len + rows]));
    }
    let mut run_batch = |eager: bool| -> Result<Vec<f32>> {
        clear_rings(backend)?;
        let (family, mut cache) = backend.prefix_cache(prefix_args, slots)?;
        let mut placements = Vec::with_capacity(seq_inputs.len());
        for (slot, &(prompt, cont)) in seq_inputs.iter().enumerate() {
            let total = prompt.len() + cont.len();
            let mut placement = cache.admit_cold(&family, prompt.len(), total + 1,
                |pages| Placement { pages, slot, ..Default::default() })?.placement;
            backend.prefill_device(&mut placement, prompt, 1, cache.pool())?;
            placements.push(placement);
        }
        backend.diagnostic_sync()?;
        let mut batch: Vec<(&mut Placement, &[u32])> = placements.iter_mut()
            .zip(seq_inputs.iter().map(|&(_, cont)| cont))
            .collect();
        let logits = if eager {
            backend.verify_eager(&mut batch, true, super::backend::verify_kind(true))?
        } else {
            backend.verify_device(&mut batch, true, cache.pool())?
        }.context("Muse wide decode check produced no logits")?;
        if !eager && backend.diagnostic_graph_width().is_some() {
            backend.diagnostic_require_graph_rows(wide)?;
        }
        logits.to_host(backend.library())
    };
    let mut outputs = Vec::new();
    for eager in [false, false, true] {
        outputs.push(run_batch(eager)?);
    }
    let bits: Vec<Vec<u32>> = outputs.iter().map(|v| v.iter().map(|x| x.to_bits()).collect()).collect();
    ensure!(same_logit_bits(&bits[0], &bits[1]) && same_logit_bits(&bits[1], &bits[2]),
        "Muse wide {wide}-row graph capture/replay/eager logits differ");
    println!("Muse wide decode graph capture/replay/eager: {wide} rows ({counts:?}), every logit bit identical");
    let vocab = backend.cfg.vocab_size;
    let wide_logits = &outputs[2];
    let mut offset = 0usize;
    let mut top1_matches = 0usize;
    let mut kl_sum = 0.0f64;
    for &(prompt, cont) in &seq_inputs {
        clear_rings(backend)?;
        let (family, mut cache) = backend.prefix_cache(prefix_args, slots)?;
        let mut placement = cache.admit_cold(&family, prompt.len(), prompt.len() + cont.len() + 1,
            |pages| Placement { pages, slot: 0, ..Default::default() })?.placement;
        backend.prefill_device(&mut placement, prompt, 1, cache.pool())?;
        backend.diagnostic_sync()?;
        let solo = backend.verify_eager(&mut [(&mut placement, cont)], true, super::backend::verify_kind(true))?
            .context("Muse wide solo control produced no logits")?
            .to_host(backend.library())?;
        for r in 0..cont.len() {
            let a = &solo[r * vocab..(r + 1) * vocab];
            let b = &wide_logits[(offset + r) * vocab..(offset + r + 1) * vocab];
            let argmax = |slice: &[f32]| slice.iter().enumerate()
                .max_by(|(_, x), (_, y)| x.total_cmp(y)).map(|(idx, _)| idx).unwrap_or(0);
            if argmax(a) == argmax(b) { top1_matches += 1; }
            let max_a = a.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let max_b = b.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let log_za = a.iter().map(|&v| ((v as f64) - max_a).exp()).sum::<f64>().ln() + max_a;
            let log_zb = b.iter().map(|&v| ((v as f64) - max_b).exp()).sum::<f64>().ln() + max_b;
            let mut row_kl = 0.0f64;
            for (&va, &vb) in a.iter().zip(b.iter()) {
                let la = va as f64 - log_za;
                let lb = vb as f64 - log_zb;
                row_kl += la.exp() * (la - lb);
            }
            kl_sum += row_kl.max(0.0);
        }
        offset += cont.len();
    }
    let top1 = top1_matches as f64 / wide as f64;
    let mean_kl = kl_sum / wide as f64;
    println!("Muse wide {wide}-row vs solo-sequence eager: top1={top1:.6} ({top1_matches}/{wide}), mean_kl={mean_kl:.6}");
    ensure!(top1 >= 0.99 && mean_kl <= 0.005,
        "Muse wide {wide}-row verify batch drifted from solo eager: top1={top1:.6}, mean_kl={mean_kl:.6}");
    Ok(())
}

fn singleton(backend: &MuseBackend<'_, '_>, s: &mut PrefillSeq<'_, Placement, DraftState>, pool: &RefPagePool)
    -> Result<Vec<f32>> {
    let logits = backend.prefill_device(s.placement, s.tokens, 1, pool)?.context("Muse singleton head")?
        .row_host(backend.library(), 0)?;
    if s.neural { backend.prefill_draft(s.placement, s.start, s.tokens, s.next, s.draft)?; }
    Ok(logits)
}

fn packed_unit(backend: &MuseBackend<'_, '_>, sequences: &mut [PrefillSeq<'_, Placement, DraftState>],
    pool: &RefPagePool,
    callback: &mut dyn FnMut(usize, crate::shared::token_io::DeviceLogits)
        -> crate::shared::prefill_packing::MemberOutcome) -> Result<Vec<Result<()>>> {
    ensure!(sequences.len() > 1, "Muse packed diagnostic requires multiple members");
    let rows = sequences.iter().map(|sequence| sequence.tokens.len()).collect::<Vec<_>>();
    let outcomes = backend.prefill_packed(sequences, pool, callback)?;
    println!("Muse packed unit completed: members {}, member rows {rows:?}, total rows {}",
        sequences.len(), rows.iter().sum::<usize>());
    Ok(outcomes)
}

fn clear_rings(backend: &MuseBackend<'_, '_>) -> Result<()> {
    backend.diagnostic_sync()?;
    // Bound host scratch even for the max-context gate; untouched ring rows
    // must have the same sentinel in singleton and packed diagnostic slots.
    let zero = vec![0; 1 << 20];
    for rank in backend.ranks {
        rank.device.run(|| {
            let ring = rank.kv.rings.buffer;
            for at in (0..ring.bytes).step_by(zero.len()) {
                let bytes = (ring.bytes - at).min(zero.len());
                backend.library().copy_h2d(view(ring, at, bytes), &zero[..bytes])?;
            }
            Ok(())
        })?;
    }
    Ok(())
}

fn read(backend: &MuseBackend<'_, '_>, stream: *mut std::ffi::c_void, buffer: Buffer) -> Result<Vec<u8>> {
    let mut bytes = vec![0; buffer.bytes];
    // SAFETY: `stream` owns this rank's work on the current device and remains live.
    unsafe { backend.library().copy_d2h(&mut bytes, buffer, stream)?; }
    Ok(bytes)
}

fn view(buffer: Buffer, offset: usize, bytes: usize) -> Buffer {
    Buffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer }
}

/// Compare each path's live continuation with its own restored prompt-end state.
/// Packed/singleton arithmetic drift remains the shared pooled gate's concern.
fn snapshot_and_restore(backend: &MuseBackend<'_, '_>, family: &super::prefix::MusePrefix<'_, '_>,
    placement: &Placement, restore_slot: usize, pool: &RefPagePool, continuation: &[u32]) -> Result<Vec<u8>> {
    let boxed = |error: cuteafd_engine::prefix::BoxError| anyhow::anyhow!(error.to_string());
    backend.diagnostic_sync()?;
    family.capture(MarkSlot(0), placement, placement.len).map_err(boxed)?;
    family.drain().map_err(boxed)?;
    let mut restored = placement.clone();
    restored.slot = restore_slot;
    restored.len = 0;
    family.restore(Some(MarkSlot(0)), &mut restored, placement.len).map_err(boxed)?;
    family.capture(MarkSlot(1), &restored, placement.len).map_err(boxed)?;
    family.drain().map_err(boxed)?;
    let cache = cuteafd_loader::families::muse_spark::cache::MuseCacheLayout::with_format(backend.cfg, backend.ranks.len(), backend.ranks[0].kv.format)?;
    let record = usize::try_from(cache.record_bytes)?;
    let ring_bytes = usize::try_from(cache.ring.rows)? * record;
    let before = family.mark_segments(MarkSlot(0));
    let after = family.mark_segments(MarkSlot(1));
    let mut snapshot = Vec::new();
    for (rank_index, rank) in backend.ranks.iter().enumerate() {
        rank.device.run(|| {
            let mark = |range: cuteafd_hostcache::copy::DeviceRange| Buffer {
                ptr: range.addr as *mut std::ffi::c_void, bytes: range.bytes, device_id: rank.device.id, flags: 0 };
            let stream = backend.streams[rank_index];
            let a = read(backend, stream, mark(before[rank_index]))?;
            let b = read(backend, stream, mark(after[rank_index]))?;
            let mark_layer_bytes = backend.cfg.window * record;
            let visible_bytes = placement.len.min(backend.cfg.window) * record;
            // Short prompts initialize only their visible rows, not mark padding.
            for (a, b) in a.chunks_exact(mark_layer_bytes).zip(b.chunks_exact(mark_layer_bytes)) {
                ensure!(a[..visible_bytes] == b[..visible_bytes],
                    "Muse prompt-end restore changed the visible ring/mark on rank {rank_index}");
                snapshot.extend_from_slice(&a[..visible_bytes]);
            }
            for layer in &rank.kv.layers {
                match *layer {
                    super::device::LayerKv::Paged { cache } => {
                        for (i, &page) in placement.pages.iter().enumerate() {
                            let rows = placement.len.saturating_sub(i * 64).min(64);
                            if rows > 0 { snapshot.extend(read(backend, stream, view(cache, page as usize * 64 * record, rows * record))?); }
                        }
                    }
                    super::device::LayerKv::Ring { ring } => {
                        snapshot.extend(read(backend, stream, view(ring, placement.slot * ring_bytes, ring_bytes))?);
                    }
                }
            }
            Ok(())
        })?;
    }
    // Preserve prompt-end evidence before either continuation mutates rings/pages.
    // Full pages alias only the saved prefix; each replay overwrites its own new rows
    // before attention reads them, including a partial tail page.
    let mut live = placement.clone();
    let live_bits = continuation_bits(backend, &mut live, continuation, pool)?;
    backend.diagnostic_sync()?;
    restored.len = 0;
    family.restore(Some(MarkSlot(0)), &mut restored, placement.len).map_err(boxed)?;
    family.drain().map_err(boxed)?;
    let restored_bits = continuation_bits(backend, &mut restored, continuation, pool)?;
    ensure!(same_logit_bits(&live_bits, &restored_bits),
        "Muse cache continuation differs from its own live prefill: slot {} -> {}, {} rows, {} decode tokens",
        placement.slot, restore_slot, placement.len, continuation.len());
    println!("Muse prompt-end restore: slot {} -> {}, {} rows, visible SWA marks and {}-token continuation logits byte-identical",
        placement.slot, restore_slot, placement.len, continuation.len());
    if let Some(max_width) = backend.diagnostic_graph_width() {
        for (rows, speculative) in graph_oracle_cases(max_width) {
            let tokens = continuation.iter().copied().cycle().take(rows).collect::<Vec<_>>();
            let mut outputs = Vec::new();
            // Repeat the graph arm from exactly the same slot/frontier to test
            // replay as well as first capture, then bypass capture for the control.
            for eager in [false, false, true] {
                backend.diagnostic_sync()?;
                restored.len = 0;
                family.restore(Some(MarkSlot(0)), &mut restored, placement.len).map_err(boxed)?;
                family.drain().map_err(boxed)?;
                let sequences = &mut [(&mut restored, tokens.as_slice())];
                let logits = if eager {
                    backend.verify_eager(sequences, speculative, super::backend::verify_kind(speculative))?
                } else {
                    backend.verify_device(sequences, speculative, pool)?
                }.context("Muse graph oracle produced no logits")?;
                outputs.push(logits.to_host(backend.library())?.iter().map(|value| value.to_bits()).collect::<Vec<_>>());
                if !eager { backend.diagnostic_require_graph_rows(rows)?; }
            }
            ensure!(same_logit_bits(&outputs[0], &outputs[1]) && same_logit_bits(&outputs[1], &outputs[2]),
                "Muse graph capture/replay/eager logits differ: {rows} rows, speculative={speculative}");
            println!("Muse graph capture/replay/eager: {rows} rows, speculative={speculative}, every logit bit identical");
        }
    }
    Ok(snapshot)
}

fn continuation_bits(backend: &MuseBackend<'_, '_>, placement: &mut Placement, tokens: &[u32], pool: &RefPagePool)
    -> Result<Vec<u32>> {
    let mut bits = Vec::new();
    for token in tokens {
        let logits = backend.verify_device(&mut [(&mut *placement, std::slice::from_ref(token))], false, pool)?
            .context("Muse cache continuation produced no logits")?.row_host(backend.library(), 0)?;
        bits.extend(logits.iter().map(|value| value.to_bits()));
    }
    Ok(bits)
}

fn graph_oracle_cases(max_width: usize) -> Vec<(usize, bool)> {
    let mut cases = vec![(1, false)];
    let mut widths = [2, 4, max_width].into_iter().filter(|&width| width > 0 && width <= max_width)
        .collect::<Vec<_>>();
    widths.sort_unstable();
    widths.dedup();
    cases.extend(widths.into_iter().map(|width| (width + 1, true)));
    cases
}

fn same_logit_bits(live: &[u32], restored: &[u32]) -> bool {
    !live.is_empty() && live == restored
}

#[cfg(test)]
mod tests {
    use super::{graph_oracle_cases, same_logit_bits, wide_verify_partition};

    #[test]
    fn wide_verify_partition_stays_within_swa_rollback_step() {
        assert_eq!(wide_verify_partition(128), [32, 32, 32, 32]);
        assert_eq!(wide_verify_partition(96), [32, 32, 32]);
        assert_eq!(wide_verify_partition(65), [22, 22, 21]);
        for total in [65, 80, 96, 128] {
            let parts = wide_verify_partition(total);
            assert_eq!(parts.iter().sum::<usize>(), total);
            assert!(parts.iter().all(|&n| n >= 1 && n <= 64));
        }
    }

    #[test]
    fn graph_oracle_covers_decode_and_anchor_plus_draft_widths() {
        assert_eq!(graph_oracle_cases(15), [(1, false), (3, true), (5, true), (16, true)]);
        assert_eq!(graph_oracle_cases(4), [(1, false), (3, true), (5, true)]);
        assert_eq!(graph_oracle_cases(1), [(1, false), (2, true)]);
        assert_eq!(graph_oracle_cases(0), [(1, false)]);
    }

    #[test]
    fn cache_continuation_checks_every_float_bit_and_row() {
        let bits = [1.0f32.to_bits(), 0.0f32.to_bits(), (-3.0f32).to_bits()];
        assert!(same_logit_bits(&bits, &bits));
        for at in 0..bits.len() {
            let mut changed = bits;
            changed[at] ^= 1;
            assert!(!same_logit_bits(&bits, &changed));
        }
        assert!(!same_logit_bits(&[0.0f32.to_bits()], &[(-0.0f32).to_bits()]));
        assert!(!same_logit_bits(&bits, &bits[..2]));
        assert!(!same_logit_bits(&[], &[]));
    }
}
