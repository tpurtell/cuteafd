//! Muse's official aspect-ratio candidate grid, including CPython set tie order.
//!
//! The reference inserts at most four nonnegative integer pairs in an eight-slot
//! set. Hashing is seed-independent; reproducing that bounded table preserves
//! `min`'s first-on-tie result without a Python dependency at serving time.
pub fn resize_shape(height: u32, width: u32, factor: u32, tokens: u32) -> Option<(u32, u32)> {
    if height == 0 || width == 0 || factor == 0 || tokens == 0 { return None; }
    let mut h = f64::from(height) / f64::from(factor);
    let mut w = f64::from(width) / f64::from(factor);
    let ratio = w / h;
    if h * w > f64::from(tokens) {
        h = (f64::from(tokens) / ratio).sqrt();
        w = h * ratio;
    }
    let mut table = [None; 8];
    for ch in [h.floor() as u64, h.ceil() as u64] {
        for cw in [w.floor() as u64, w.ceil() as u64] {
            let mut hash = 2_870_177_450_012_600_261u64;
            for lane in [ch, cw] {
                hash = hash.wrapping_add(lane.wrapping_mul(14_029_467_366_897_019_727));
                hash = hash.rotate_left(31).wrapping_mul(11_400_714_785_074_694_791);
            }
            hash = hash.wrapping_add(2 ^ (2_870_177_450_012_600_261 ^ 3_527_539));
            if hash == u64::MAX { hash = 1_546_275_796; }
            let mut perturb = hash;
            let mut slot = hash as usize & 7;
            loop {
                match table[slot] {
                    Some(pair) if pair == (ch, cw) => break,
                    None => { table[slot] = Some((ch, cw)); break; }
                    _ => { perturb >>= 5; slot = (slot * 5 + 1 + perturb as usize) & 7; }
                }
            }
        }
    }
    let mut best = None;
    let mut error = f64::INFINITY;
    for (ch, cw) in table.into_iter().flatten() {
        if ch == 0 || cw == 0 || ch * cw > u64::from(tokens) { continue; }
        let candidate_error = ((ch as f64 / cw as f64) - f64::from(height) / f64::from(width)).abs();
        if candidate_error < error { best = Some((ch, cw)); error = candidate_error; }
    }
    let (ch, cw) = best.unwrap_or_else(|| (h.round_ties_even().max(1.0) as u64, w.round_ties_even().max(1.0) as u64));
    Some((u32::try_from(ch.checked_mul(u64::from(factor))?).ok()?,
        u32::try_from(cw.checked_mul(u64::from(factor))?).ok()?))
}
