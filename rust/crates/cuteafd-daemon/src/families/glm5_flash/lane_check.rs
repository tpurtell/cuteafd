//! glmf-golden --lane-check: N prefill lanes give the bits of N passes. One chunk prefills through
//! the pipelined lanes into sequence A and, over the same cuts, through serial passes into
//! sequence B; every row's logits, the KDA state (with everything `slot_state` carries, the compact
//! index cache's tails included) and every paged byte of both sequences' units must agree bit for
//! bit. Both sequences take fresh units of a fresh engine, which the caches zero, so whole units
//! compare exactly whatever their layout (with or without the per-token index keys).
use super::engine::{Allocator, GlmfEngine, GlmfPlacement};
use anyhow::{ensure, Context, Result};

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

/// Prefills `tokens` (up to one pipelined chunk) both ways and compares them.
pub(crate) fn lane_check(engine: &GlmfEngine<'_>, tokens: &[u32]) -> Result<()> {
    ensure!(engine.weights.layers.len() == engine.cfg.layers, "--lane-check needs every layer");
    ensure!(engine.full_prefill_logits, "--lane-check needs every prefill row's logits");
    let n = tokens.len().min(engine.prefill_capacity());
    let cuts = engine.prefill_cuts(n)?;
    ensure!(engine.prefill_capacity() > engine.prefill_rows || engine.prefill_lane_count == 1,
        "--lane-check needs admitted pipelined prefill lanes (every layer)");
    let mut allocator = Allocator::for_engine(engine);
    let (mut lanes, mut serial) = (allocator.admit(n)?, allocator.admit(n)?);
    let started = std::time::Instant::now();
    let lane_logits = engine.prefill_forced(&mut lanes, &tokens[..n], None, None, true)?
        .context("--lane-check needs the head")?;
    let lane_seconds = started.elapsed().as_secs_f64();
    let started = std::time::Instant::now();
    let mut serial_logits = Vec::with_capacity(lane_logits.len());
    let mut first = 0;
    for &rows in &cuts {
        serial_logits.extend(engine.prefill_serial(&mut serial, &tokens[first..first + rows], true)?
            .context("--lane-check needs the head")?);
        first += rows;
    }
    let serial_seconds = started.elapsed().as_secs_f64();
    let logits = lane_logits.len() == serial_logits.len()
        && lane_logits.iter().zip(&serial_logits).all(|(a, b)| a.to_bits() == b.to_bits());
    let state = engine.slot_state(lanes.slot)? == engine.slot_state(serial.slot)?;
    let paged = unit_bytes(engine, &lanes)? == unit_bytes(engine, &serial)?;
    let verdict = |same: bool| if same { "identical" } else { "DIFFER" };
    println!("lane check: {n} tokens in {} lanes of up to {} rows (cuts {cuts:?}): logits {}, KDA state {}, paged \
        units {} (lanes {:.0} ms, serial passes {:.0} ms)", engine.prefill_lane_count, engine.prefill_rows,
        verdict(logits), verdict(state), verdict(paged), 1e3 * lane_seconds, 1e3 * serial_seconds);
    allocator.release(serial);
    allocator.release(lanes);
    ensure!(logits && state && paged, "the lanes differ from serial passes over the same cuts");
    Ok(())
}

/// Context units live on one rank; whole/head-owned layers use the global unit ID.
fn local_unit(global: u32, rank: usize, context: bool) -> Option<u32> {
    if !context { Some(global) } else { ((global & 1) as usize == rank).then_some(global / 2) }
}

#[cfg(test)]
mod tests {
    use super::*;
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
