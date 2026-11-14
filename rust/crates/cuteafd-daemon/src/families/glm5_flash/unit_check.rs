//! Same-cut executor serial/admitted-depth golden and shared token/cache diagnostics.
use super::engine::{Allocator, GlmfEngine, GlmfPlacement};
use anyhow::{ensure, Context, Result};

/// Token-only diagnostics may consume the shared, checksum-validated quick fidelity windows.
pub(crate) fn input_tokens(root: &std::path::Path, vocab: usize) -> Result<Vec<u32>> {
    let path = root.join("tokens.bin");
    let tokens: Vec<u32> = if path.is_file() {
        let bytes = std::fs::read(path)?;
        ensure!(bytes.len() % 4 == 0, "golden token bytes must be u32-aligned");
        bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
    } else {
        use cuteafd_bench::fidelity_dataset;
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(manifest["config"] == fidelity_dataset::GLMF_CONFIG && manifest["vocab"].as_u64() == Some(vocab as u64),
            "Flash golden fidelity config or vocabulary differs");
        let reference = fidelity_dataset::load(root, &manifest)?;
        let windows = fidelity_dataset::quick_windows(&reference)?;
        ensure!(windows.iter().all(|w| w.media.is_empty()), "Flash unit golden requires text-only windows");
        println!("unit golden input: sealed quick windows {:?}; set {} (concatenated only for same-cut execution identity)",
            windows.iter().map(|w| &w.id).collect::<Vec<_>>(), reference.set_sha256);
        windows.into_iter().flat_map(|w| w.tokens).collect()
    };
    ensure!(!tokens.is_empty() && tokens.iter().all(|&t| (t as usize) < vocab), "invalid golden tokens");
    Ok(tokens)
}

/// Every paged byte of `placement`'s units, unit by unit, buffer by buffer (each paged buffer
/// holds one equal slice per allocation unit).
pub(crate) fn unit_bytes(engine: &GlmfEngine<'_>, placement: &GlmfPlacement) -> Result<Vec<u8>> {
    engine.synchronize()?;
    let mut out = Vec::new();
    for rank in 0..engine.ranks() {
        engine.on(rank, || {
            for layer in engine.paged_buffers_on(rank) {
                for buffer in std::iter::once(layer.records).chain(layer.keys).chain([layer.pools]) {
                    let units = engine.local_units();
                    ensure!(units > 0 && buffer.bytes % units == 0, "a paged buffer that is not whole units");
                    let per_unit = buffer.bytes / units;
                    for &global in &placement.units {
                        let Some(unit) = local_unit(global, rank, engine.context_kv()) else { continue };
                        let unit = crate::shared::prefix::view(buffer, unit as usize * per_unit, per_unit)?;
                        let mut bytes = vec![0u8; per_unit];
                        engine.library.copy_d2h(&mut bytes, unit)?;
                        out.extend(bytes);
                    }
                }
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// Compare two shared-executor schedules with identical cuts, weights and fresh pages.
pub(crate) fn unit_check(engine: &GlmfEngine<'_>, tokens: &[u32]) -> Result<()> {
    ensure!(engine.weights.layers.len() == engine.cfg.layers, "--unit-check needs every layer");
    ensure!(engine.prefill_waves >= 2, "--unit-check needs at least two admitted slots");
    let depth = engine.prefill_waves;
    let n = unit_check_rows(tokens.len(), engine.prefill_rows, depth, engine.max_context)?;
    let cuts = engine.prefill_cuts(n)?;
    ensure!(cuts.len() > depth, "--unit-check must reuse an admitted slot");
    let mut allocator = Allocator::for_engine(engine);
    let (mut one, mut two) = (allocator.admit(n)?, allocator.admit(n)?);
    let run = |placement: &mut GlmfPlacement, waves| -> Result<([u8; 32], usize)> {
        use sha2::{Digest, Sha256};
        let (mut digest, mut rows) = (Sha256::new(), 0);
        let consumer = crate::shared::probe::CallbackRows::new(|position, values: &[f32]| {
            ensure!(position == rows + 1 && values.len() == engine.cfg.vocab_size
                && values.iter().all(|v| v.is_finite()), "unit golden row order/shape/finite mismatch");
            for value in values { digest.update(value.to_le_bytes()); }
            rows += 1;
            Ok(())
        });
        engine.prefill_unit_width(placement, &tokens[..n], waves,
            crate::shared::probe::ScoringRows { consumer: &consumer, position: 1 })?
            .context("unit golden needs the head")?;
        drop(consumer);
        ensure!(rows == n, "unit golden row coverage differs");
        Ok((digest.finalize().into(), rows))
    };
    let started = std::time::Instant::now();
    let one_logits = run(&mut one, 1)?;
    let one_seconds = started.elapsed().as_secs_f64();
    let started = std::time::Instant::now();
    let two_logits = run(&mut two, depth)?;
    let two_seconds = started.elapsed().as_secs_f64();
    let logits = one_logits == two_logits;
    let state = engine.slot_state(one.slot)? == engine.slot_state(two.slot)?;
    let paged = unit_bytes(engine, &one)? == unit_bytes(engine, &two)?;
    let verdict = |same: bool| if same { "identical" } else { "DIFFER" };
    println!("unit check: {n} tokens (same cuts {cuts:?}): logits {}, KDA state {}, paged units {} \
        (N1 {:.0} ms, N{depth} {:.0} ms)", verdict(logits), verdict(state), verdict(paged),
        1e3 * one_seconds, 1e3 * two_seconds);
    check_identity(logits, state, paged, depth)?;
    allocator.release(two);
    allocator.release(one);
    boundary_check(engine, &mut allocator, &tokens[..n])
}

/// Reuse the same peer workspace across two calls, with no host reader between
/// the candidate calls. The control explicitly drains both ranks at the boundary.
fn boundary_check(engine: &GlmfEngine<'_>, allocator: &mut Allocator, tokens: &[u32]) -> Result<()> {
    let (mut drained, mut queued) = (allocator.admit(tokens.len())?, allocator.admit(tokens.len())?);
    let split = engine.prefill_rows.min(tokens.len() - 1);
    engine.prefill_device(&mut drained, &tokens[..split])?.context("boundary control head")?;
    engine.synchronize()?;
    let expected = engine.prefill_device(&mut drained, &tokens[split..])?
        .context("boundary control continuation head")?.to_host(engine.library)?;
    engine.synchronize()?;
    engine.prefill_device(&mut queued, &tokens[..split])?.context("boundary queued head")?;
    let actual = engine.prefill_device(&mut queued, &tokens[split..])?
        .context("boundary queued continuation head")?.to_host(engine.library)?;
    let logits = expected.len() == actual.len()
        && expected.iter().zip(&actual).all(|(a, b)| a.to_bits() == b.to_bits());
    let state = engine.slot_state(drained.slot)? == engine.slot_state(queued.slot)?;
    let paged = unit_bytes(engine, &drained)? == unit_bytes(engine, &queued)?;
    println!("back-to-back prefill boundary: logits {logits}, KDA state {state}, paged units {paged}");
    ensure!(logits && state && paged, "queued prefill boundary differs from explicitly drained control");
    allocator.release(queued);
    allocator.release(drained);
    Ok(())
}

fn check_identity(logits: bool, state: bool, paged: bool, depth: usize) -> Result<()> {
    ensure!(logits && state && paged, "executor N1 and admitted N{depth} differ over the same cuts");
    Ok(())
}

/// The diagnostic may exceed one serving span: it needs a slot's second use,
/// not just one fill of the admitted ring. Serving's span limit is unchanged.
fn unit_check_rows(available: usize, unit_rows: usize, depth: usize, context: usize) -> Result<usize> {
    ensure!(depth >= 2 && unit_rows > 0, "--unit-check needs at least two admitted slots");
    let needed = unit_rows.checked_mul(depth).and_then(|n| n.checked_add(1))
        .context("unit-check row count overflow")?;
    ensure!(available >= needed && context >= needed,
        "--unit-check needs at least {needed} tokens and context rows to reuse depth {depth}");
    Ok(available.min(context).min(needed.next_multiple_of(256)))
}

/// Context units live on one rank; whole/head-owned layers use the global unit ID.
fn local_unit(global: u32, rank: usize, context: bool) -> Option<u32> {
    if !context { Some(global) } else { ((global & 1) as usize == rank).then_some(global / 2) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admitted_depth_check_requires_real_slot_reuse() -> Result<()> {
        for depth in [2, 3, 4] {
            let rows = cuteafd_loader::serving_capacity::prefill_waves::unit_rows(depth as u64) as usize;
            let available = (depth + 1) * rows;
            let n = unit_check_rows(available, rows, depth, 1 << 20)?;
            let cuts = crate::shared::prefill_pipeline::prefill_units(n, rows, depth, 256)?;
            assert!(cuts.len() > depth);
            // Negative control: the old serving-span clamp never reuses a slot.
            assert!(unit_check_rows(depth * rows, rows, depth, 1 << 20).is_err());
            assert!(unit_check_rows(available, rows, depth, depth * rows).is_err());
        }
        Ok(())
    }

    #[test]
    fn admitted_depth_oracle_rejects_each_corrupted_component() -> Result<()> {
        check_identity(true, true, true, 3)?;
        for identity in [(false, true, true), (true, false, true), (true, true, false)] {
            assert!(check_identity(identity.0, identity.1, identity.2, 3).is_err());
        }
        // Equal floating values still differ when the stored bits differ.
        assert!(check_identity(0.0f32.to_bits() == (-0.0f32).to_bits(), true, true, 3).is_err());
        Ok(())
    }

    #[test]
    fn token_only_diagnostics_reject_truncated_or_out_of_vocabulary_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        std::fs::write(&path, [1, 0, 0]).unwrap();
        assert!(input_tokens(dir.path(), 10).is_err());
        std::fs::write(&path, 10u32.to_le_bytes()).unwrap();
        assert!(input_tokens(dir.path(), 10).is_err());
        std::fs::write(&path, 9u32.to_le_bytes()).unwrap();
        assert_eq!(input_tokens(dir.path(), 10).unwrap(), vec![9]);
    }

    #[test]
    fn diagnostics_visit_each_owner_local_unit_once() {
        for global in 0..8 {
            assert_eq!(local_unit(global, 0, false), Some(global));
            assert_eq!(local_unit(global, 1, false), Some(global));
            let owners: Vec<_> = (0..2).filter_map(|rank| local_unit(global, rank, true)
                .map(|local| (rank, local))).collect();
            assert_eq!(owners, [((global & 1) as usize, global / 2)]);
        }
    }
}
