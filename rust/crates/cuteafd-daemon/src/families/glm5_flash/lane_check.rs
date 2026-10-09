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
    for layer in engine.paged_buffers() {
        // Records, token keys (`--index-cache keys` only) and pool keys.
        for buffer in std::iter::once(layer.records).chain(layer.keys).chain([layer.pools]) {
            ensure!(buffer.bytes % engine.pool_pages == 0, "a paged buffer that is not whole units");
            let per_unit = buffer.bytes / engine.pool_pages;
            for &unit in &placement.units {
                let mut bytes = vec![0u8; per_unit];
                let unit = cuteafd_ffi::CuteafdDeviceBuffer {
                    // SAFETY: unit < pool pages, so the slice lies inside the buffer.
                    ptr: unsafe { buffer.ptr.cast::<u8>().add(unit as usize * per_unit) }.cast(),
                    bytes: per_unit,
                    ..buffer
                };
                engine.library.copy_d2h(&mut bytes, unit)?;
                out.extend(bytes);
            }
        }
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
        "--lane-check needs pipelined Spark prefill lanes (--peers, every layer)");
    let mut allocator = Allocator::new(engine.pages, engine.slots);
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
