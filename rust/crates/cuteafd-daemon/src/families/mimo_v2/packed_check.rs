//! Mixed-request packing and partner-isolation golden; uses the serving executor.
use super::engine::{Allocator, MimoEngine, MimoPlacement};
use crate::shared::prefill_packing::{self, Limits};
use anyhow::{ensure, Context, Result};

const LENGTHS: [usize; 10] = [29, 40, 7, 64, 65, 127, 1, 129, 255, 100];
const PREFIX: usize = 300;

fn geometry(members: usize, limits: &Limits) -> Result<Vec<(usize, usize)>> {
    ensure!((2..=LENGTHS.len()).contains(&members), "--packed-check takes 2 to 10 sequences");
    let mut chunks = LENGTHS[..members].iter().map(|&rows| (0, rows)).collect::<Vec<_>>();
    chunks[members - 1] = (PREFIX, 100);
    for chunk in &chunks {
        prefill_packing::plan(&[*chunk], &Limits { rows: limits.chunk_rows, ..*limits })
            .context("--packed-check member exceeds admitted rows/taps/context")?;
    }
    Ok(chunks)
}

fn prompt(tokens: &[u32], member: usize, rows: usize) -> Vec<u32> {
    (0..rows).map(|row| tokens[(member * 97 + row) % tokens.len()]).collect()
}

struct Observed {
    logits: Vec<f32>,
    kv: Vec<u8>,
    frontier: usize,
}

/// Initialize all allocated pages and rings identically so unused tail bytes are
/// part of the byte comparison rather than allocator contents. Diagnostic only.
fn clear(engine: &MimoEngine<'_>, placement: &MimoPlacement) -> Result<()> {
    for rank in 0..engine.ranks() {
        engine.on(rank, || {
            // SAFETY: all prior submissions have their matching peer pushes queued.
            unsafe { engine.library.cuda_stream_synchronize(engine.stream_of(rank))?; }
            for layer in 0..engine.weights.layers.len() {
                let (attention, buffer, record) = engine.kv_layer_on(rank, layer);
                let ranges = match attention {
                    cuteafd_loader::families::mimo_v2::MimoAttention::Full => placement.pages.iter()
                        .map(|&page| (page as usize * super::engine::PAGE_ROWS, super::engine::PAGE_ROWS)).collect::<Vec<_>>(),
                    cuteafd_loader::families::mimo_v2::MimoAttention::Sliding =>
                        vec![(placement.ring as usize * super::engine::RING_ROWS, super::engine::RING_ROWS)],
                };
                for (row, rows) in ranges {
                    let bytes = rows * record;
                    let offset = row * record;
                    ensure!(offset + bytes <= buffer.bytes, "packed check clear exceeds KV storage");
                    let view = cuteafd_ffi::CuteafdDeviceBuffer {
                        ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer };
                    engine.library.cuda_zero_bytes(view, bytes)?;
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}

fn observe(engine: &MimoEngine<'_>, placement: &MimoPlacement, logits: Vec<f32>) -> Result<Observed> {
    // Include every byte of allocated full pages, including initialized tail rows,
    // and all surviving SWA records. kv_rows iterates every TP rank.
    let page_end = placement.pages.len() * super::engine::PAGE_ROWS;
    let mut kv = Vec::new();
    for rank in 0..engine.ranks() {
        engine.on(rank, || {
            // SAFETY: this engine owns the rank stream and its submitted readers.
            unsafe { engine.library.cuda_stream_synchronize(engine.stream_of(rank))?; }
            for layer in 0..engine.weights.layers.len() {
                let (attention, buffer, record) = engine.kv_layer_on(rank, layer);
                let ranges = match attention {
                    cuteafd_loader::families::mimo_v2::MimoAttention::Full => placement.pages.iter()
                        .map(|&page| (page as usize * super::engine::PAGE_ROWS, super::engine::PAGE_ROWS)).collect::<Vec<_>>(),
                    cuteafd_loader::families::mimo_v2::MimoAttention::Sliding =>
                        vec![(placement.ring as usize * super::engine::RING_ROWS, super::engine::RING_ROWS)],
                };
                for (row, rows) in ranges {
                    let bytes = rows * record;
                    let offset = row * record;
                    ensure!(offset + bytes <= buffer.bytes, "packed check read exceeds KV storage");
                    let first = kv.len();
                    kv.resize(first + bytes, 0);
                    // SAFETY: `engine.stream_of(rank)` owns rank's KV writes on this device.
                    unsafe { engine.library.copy_d2h(&mut kv[first..], cuteafd_ffi::CuteafdDeviceBuffer {
                        ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer }, engine.stream_of(rank))?; }
                }
            }
            Ok(())
        })?;
    }
    ensure!(placement.len <= page_end, "packed frontier exceeds allocated pages");
    Ok(Observed { logits, kv, frontier: placement.len })
}

fn arm(engine: &MimoEngine<'_>, prompts: &[Vec<u32>], chunks: &[(usize, usize)], packed: bool)
    -> Result<Vec<Observed>> {
    let mut allocator = Allocator::new(engine.pages, engine.rings);
    let mut placements = prompts.iter().map(|p| allocator.admit(p.len())).collect::<Result<Vec<_>>>()?;
    for ((placement, prompt), &(start, _)) in placements.iter_mut().zip(prompts).zip(chunks) {
        clear(engine, placement)?;
        for part in prompt[..start].chunks(engine.prefill_capacity()) {
            engine.prefill_device(placement, part, false, None, None)?;
        }
    }
    let mut logits = vec![Vec::new(); placements.len()];
    if packed {
        let limits = engine.packed_limits();
        let groups = prefill_packing::groups(chunks.len(), |range|
            prefill_packing::plan(&chunks[range], &limits).is_ok());
        for range in groups {
            if range.len() == 1 {
                let i = range.start;
                logits[i] = engine.prefill_device(&mut placements[i], &prompts[i][chunks[i].0..], false, None, None)?
                    .context("packed check requires all layers")?.row_host(engine.library, 0)?;
                continue;
            }
            let first = range.start;
            let mut members = placements[range.clone()].iter_mut().zip(&prompts[range.clone()]).zip(&chunks[range])
                .map(|((p, tokens), &(start, _))| (p, &tokens[start..])).collect::<Vec<_>>();
            let (_, outcomes) = engine.prefill_packed(&mut members, &mut |i, rows| {
                logits[first + i] = rows.row_host(engine.library, 0)?;
                Ok(Ok(()))
            })?;
            for outcome in outcomes { outcome?; }
        }
    } else {
        for (i, ((p, tokens), &(start, _))) in placements.iter_mut().zip(prompts).zip(chunks).enumerate() {
            logits[i] = engine.prefill_device(p, &tokens[start..], false, None, None)?
                .context("packed check requires all layers")?.row_host(engine.library, 0)?;
        }
    }
    placements.iter().zip(logits).map(|(p, logits)| observe(engine, p, logits)).collect()
}

fn same_logits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[derive(Debug)]
struct LogitDifference {
    rows: usize,
    max_abs: f64,
    mean_kl: f64,
    top1_agree: usize,
}

fn logit_difference(packed: &[f32], alone: &[f32], vocab: usize) -> Result<LogitDifference> {
    ensure!(vocab > 0 && !packed.is_empty() && packed.len() == alone.len() && packed.len() % vocab == 0,
        "packed diagnostic requires matching nonempty whole logit rows");
    ensure!(packed.iter().chain(alone).all(|x| x.is_finite()), "packed diagnostic has nonfinite logits");
    let log_partition = |row: &[f32]| {
        let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        top + row.iter().map(|&x| (f64::from(x) - top).exp()).sum::<f64>().ln()
    };
    let argmax = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let rows = packed.len() / vocab;
    let mut diff = LogitDifference { rows, max_abs: 0.0, mean_kl: 0.0, top1_agree: 0 };
    for (ours, reference) in packed.chunks_exact(vocab).zip(alone.chunks_exact(vocab)) {
        let (ours_z, reference_z) = (log_partition(ours), log_partition(reference));
        diff.top1_agree += usize::from(argmax(ours) == argmax(reference));
        for (&ours, &reference) in ours.iter().zip(reference) {
            let (ours, reference) = (f64::from(ours), f64::from(reference));
            diff.max_abs = diff.max_abs.max((ours - reference).abs());
            let log_p = reference - reference_z;
            diff.mean_kl += log_p.exp() * (log_p - (ours - ours_z));
        }
    }
    diff.mean_kl /= rows as f64;
    Ok(diff)
}

fn partner_in_group(group: &std::ops::Range<usize>, target: usize) -> Option<usize> {
    (group.contains(&target) && group.len() > 1)
        .then(|| if target + 1 < group.end { target + 1 } else { group.start })
}

fn changed_partner(prompts: &[Vec<u32>], partner: usize, vocab_size: usize) -> Vec<Vec<u32>> {
    let mut changed = prompts.to_vec();
    for token in &mut changed[partner] { *token = ((*token as usize + 1) % vocab_size) as u32; }
    changed
}

pub(crate) fn run(engine: &MimoEngine<'_>, tokens: &[u32], members: usize) -> Result<()> {
    ensure!(!tokens.is_empty() && tokens.iter().all(|&t| (t as usize) < engine.cfg.vocab_size), "invalid packed-check tokens");
    ensure!(engine.weights.layers.len() == engine.cfg.layers, "--packed-check requires every layer");
    let limits = engine.packed_limits();
    let chunks = geometry(members, &limits)?;
    let groups = prefill_packing::groups(members, |range| prefill_packing::plan(&chunks[range], &limits).is_ok());
    println!("MiMo packed check: row cap {}, member groups {:?} (one-member groups run alone)", limits.rows, groups);
    let prompts = chunks.iter().enumerate().map(|(i, &(start, rows))| prompt(tokens, i, start + rows)).collect::<Vec<_>>();
    let packed = arm(engine, &prompts, &chunks, true)?;
    let alone = arm(engine, &prompts, &chunks, false)?;
    let mut exact = true;
    for (i, (a, b)) in packed.iter().zip(&alone).enumerate() {
        let logits = same_logits(&a.logits, &b.logits);
        let kv = a.kv == b.kv;
        let frontier = a.frontier == b.frontier && a.frontier == prompts[i].len();
        println!("MiMo packed check: member {i} rows {} start {}: logits {logits}, KV {kv}, frontier {frontier}", chunks[i].1, chunks[i].0);
        if !(logits && kv && frontier) {
            // The checker saves the terminal logit row, not every input token's row.
            let diff = logit_difference(&a.logits, &b.logits, engine.cfg.vocab_size)?;
            ensure!(a.kv.len() == b.kv.len() && !a.kv.is_empty(), "packed diagnostic KV lengths differ or are empty");
            let kv_diff = a.kv.iter().zip(&b.kv).filter(|(a, b)| a != b).count();
            println!("MiMo packed difference: member {i}: saved_rows={} max_abs={:.9e} mean_KL(alone||packed)={:.9e} top1_agreement={}/{} KV_differing_bytes={}/{} KV_diff_fraction={:.9e}",
                diff.rows, diff.max_abs, diff.mean_kl, diff.top1_agree, diff.rows,
                kv_diff, a.kv.len(), kv_diff as f64 / a.kv.len() as f64);
        }
        exact &= logits && kv && frontier;
    }
    // Exercise both a fresh target and the resumed final member at fixed geometry.
    for target in [0, members - 1] {
        let group = groups.iter().find(|group| group.contains(&target)).context("packed target group")?;
        let Some(partner_index) = partner_in_group(group, target) else {
            println!("MiMo packed check: member {target} runs alone; no packed partner to replace");
            continue;
        };
        let changed = changed_partner(&prompts, partner_index, engine.cfg.vocab_size);
        let partner = arm(engine, &changed, &chunks, true)?;
        let isolated = same_logits(&packed[target].logits, &partner[target].logits)
            && packed[target].kv == partner[target].kv
            && packed[target].frontier == partner[target].frontier;
        println!("MiMo packed check: member {target} start {} unchanged with partner replaced at fixed geometry: {isolated}", chunks[target].0);
        ensure!(isolated, "packed partner isolation failed for member {target}");
    }
    ensure!(exact, "packed differs from alone; characterize expert route changes before accepting numerical differences");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn logit_diagnostics_score_every_saved_row_with_alone_as_reference() {
        let alone = [0.0, 0.0, 4.0, 0.0];
        let packed = [4.0, 0.0, 0.0, 4.0];
        let diff = logit_difference(&packed, &alone, 2).unwrap();
        let p = 4f64.exp() / (4f64.exp() + 1.0);
        let first_kl = -2.0 + (4f64.exp() + 1.0).ln() - 2f64.ln();
        assert_eq!(diff.rows, 2);
        assert_eq!(diff.max_abs, 4.0);
        assert_eq!(diff.top1_agree, 0);
        assert!((diff.mean_kl - (first_kl + 4.0 * (2.0 * p - 1.0)) / 2.0).abs() < 1e-12);
        let identical = logit_difference(&alone, &alone, 2).unwrap();
        assert_eq!(identical.max_abs, 0.0);
        assert_eq!(identical.mean_kl, 0.0);
        assert_eq!(identical.top1_agree, 2);
        let shifted = logit_difference(&[1000.0, 1000.0, 1004.0, 1000.0], &alone, 2).unwrap();
        assert!(shifted.mean_kl.abs() < 1e-12);
        assert_eq!(shifted.top1_agree, 2);
    }

    #[test]
    fn logit_diagnostics_reject_invalid_or_nonfinite_rows() {
        for (packed, alone, vocab) in [(&[][..], &[][..], 2), (&[0.0][..], &[0.0][..], 0),
            (&[0.0][..], &[0.0][..], 2), (&[0.0][..], &[0.0, 1.0][..], 1),
            (&[f32::NAN][..], &[0.0][..], 1), (&[0.0][..], &[f32::INFINITY][..], 1)] {
            assert!(logit_difference(packed, alone, vocab).is_err());
        }
    }

    #[test]
    fn capped_check_replaces_a_partner_inside_the_resumed_targets_pack() {
        let limits = Limits { rows: 640, chunk_rows: 4096, max_context: 4096, taps: Some(1024) };
        let chunks = geometry(10, &limits).unwrap();
        let groups = prefill_packing::groups(chunks.len(), |range| prefill_packing::plan(&chunks[range], &limits).is_ok());
        assert_eq!(groups, [0..8, 8..10]);
        assert_eq!(partner_in_group(&groups[0], 0), Some(1));
        assert_eq!(partner_in_group(&groups[1], 9), Some(8));
        assert_eq!(partner_in_group(&(9..10), 9), None);
        assert_eq!(partner_in_group(&(0..8), 9), None);
    }

    #[test]
    fn partner_changes_preserve_fresh_and_resumed_target_geometry() {
        let limits = Limits { rows: 4096, chunk_rows: 4096, max_context: 4096, taps: Some(1024) };
        for members in 2..=10 {
            let chunks = geometry(members, &limits).unwrap();
            let prompts = chunks.iter().enumerate()
                .map(|(i, &(start, rows))| prompt(&[0, 1, 2, 3], i, start + rows)).collect::<Vec<_>>();
            for target in [0, members - 1] {
                let changed = changed_partner(&prompts, (target + 1) % members, 4);
                assert_eq!(changed[target], prompts[target]);
                assert_eq!(changed.iter().map(Vec::len).collect::<Vec<_>>(), prompts.iter().map(Vec::len).collect::<Vec<_>>());
                let partner = (target + 1) % members;
                for (i, (before, after)) in prompts.iter().zip(&changed).enumerate() {
                    if i == partner {
                        assert!(before.iter().zip(after).all(|(a, b)| a != b));
                    } else {
                        assert_eq!(before, after);
                    }
                }
            }
            assert_eq!(chunks[members - 1].0, PREFIX);
        }
    }

    #[test]
    fn mixed_geometry_keeps_resumed_member_and_capacity_bound() {
        let limits = Limits { rows: 4096, chunk_rows: 4096, max_context: 4096, taps: Some(1024) };
        for members in 2..=10 {
            let chunks = geometry(members, &limits).unwrap();
            assert_eq!(chunks.len(), members);
            assert_eq!(chunks[members - 1], (PREFIX, 100));
        }
        for members in [0, 1, 11] { assert!(geometry(members, &limits).is_err()); }
        let capped = Limits { rows: 128, ..limits };
        let chunks = geometry(2, &capped).unwrap();
        assert_eq!(prefill_packing::groups(chunks.len(), |range| prefill_packing::plan(&chunks[range], &capped).is_ok()),
            vec![0..1, 1..2]);
        assert!(geometry(2, &Limits { chunk_rows: 64, ..limits }).is_err());
    }
}
