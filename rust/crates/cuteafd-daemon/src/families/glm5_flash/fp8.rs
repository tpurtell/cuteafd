//! FP8 (E4M3) decode copies of GLM 5.3 Flash coordinator weights.
//!
//! Three scale layouts, all FP32 with 128-wide K blocks:
//! - `Block`: the checkpoint's 128x128 grid `[N/128, K/128]` (MLA, dense and
//!   shared experts: the official FP8 release's own E4M3 bytes and scales).
//! - `Row128`: one scale per output row and 128-wide K block, `[N, K/128]`.
//! - `Channel`: one scale per output row (W8A16), stored repeated as `[N, K/128]`.
//!
//! Quantization of a BF16 weight: `s = amax / 448` (or `--fp8-scales`' other
//! rules, [`crate::shared::fp8_linear::Fp8Scales`]), `q = e4m3_rn(w / s)`
//! (saturating), the same recipe as the checkpoint's own blocks.
use crate::shared::fp8_linear::Fp8Scales;

/// The smallest power of two >= `amax / 448` (1 for zero): Hugh Madden's
/// glm53f-afd `pow2_scale`, bit for bit the native kernels'.
pub(crate) fn pow2_scale(amax: f32) -> f32 {
    if amax <= 0.0 || amax.is_nan() {
        return 1.0;
    }
    let b = amax.to_bits();
    // amax = m 2^e with m in [1, 2): 2^(e - 8), or 2^(e - 7) when m > 1.75 (448 = 1.75 x 2^8).
    let x = ((b >> 23) as i32 - 135 + i32::from((b & 0x7F_FFFF) > 0x60_0000)).clamp(-126, 127);
    f32::from_bits(((x + 127) as u32) << 23)
}

/// E4M3 (fn: no infinities, 0x7F NaN) of `x`, round to nearest even, saturating at 448.
pub(crate) fn e4m3(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs();
    if a.is_nan() {
        return 0x7F;
    }
    // [448, 464) rounds to 448; above that it would round to 480 (NaN): saturate.
    if a >= 464.0 {
        return sign | 0x7E;
    }
    if a < 0.015625 {
        // Subnormals: multiples of 2^-9; 8 * 2^-9 = 2^-6 encodes as the first normal (0x08).
        return sign | (a * 512.0).round_ties_even() as u8;
    }
    let bits = a.to_bits();
    let rounded = bits + 0x7_FFFF + ((bits >> 20) & 1);
    let exponent = ((rounded >> 23) & 0xFF) as i32 - 127;
    let mantissa = ((rounded >> 20) & 7) as u8;
    sign | (((exponent + 7) as u8) << 3) | mantissa
}

/// The value of an E4M3 (fn) byte.
pub(crate) fn e4m3_value(q: u8) -> f32 {
    let sign = if q & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (((q >> 3) & 0xF) as i32, (q & 7) as f32);
    sign * if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}

/// Nearest E2M1 magnitude (0, 0.5, 1, 1.5, 2, 3, 4, 6; ties to the even code).
fn e2m1(a: f32) -> f32 {
    match a {
        a if a <= 0.25 => 0.0,
        a if a < 0.75 => 0.5,
        a if a <= 1.25 => 1.0,
        a if a < 1.75 => 1.5,
        a if a <= 2.5 => 2.0,
        a if a < 3.5 => 3.0,
        a if a <= 5.0 => 4.0,
        _ => 6.0,
    }
}

/// A BF16 `[n, k]` weight quantized to NVFP4 (E2M1 values, one E4M3 scale per
/// 16 values along K, no global scale) and dequantized back to BF16 (exact:
/// every E2M1 x E4M3 product fits BF16), for numerics gates before a kernel
/// exists. `search`: per group, the scale among 0.70..1.30 x amax/6 (E4M3
/// rounded) with the least squared error, instead of amax/6.
pub(crate) fn nvfp4_roundtrip(weight: &[u8], n: usize, k: usize, search: bool) -> Vec<u8> {
    assert!(weight.len() == n * k * 2 && k % 16 == 0);
    let mut out = vec![0u8; weight.len()];
    let threads = std::thread::available_parallelism().map_or(8, |p| p.get()).min(32);
    let per = n.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (chunk, first) in out.chunks_mut(per * k * 2).zip((0..n).step_by(per)) {
            scope.spawn(move || {
                for row in first..(first + per).min(n) {
                    for g in 0..k / 16 {
                        let xs: Vec<f32> = (0..16).map(|i| bf16(weight, row * k + g * 16 + i)).collect();
                        let amax = xs.iter().fold(0f32, |m, x| m.max(x.abs()));
                        let quant = |s: f32| -> (f32, Vec<f32>) {
                            let ys: Vec<f32> = xs.iter().map(|x| x.signum() * e2m1(x.abs() / s) * s).collect();
                            (xs.iter().zip(&ys).map(|(x, y)| (x - y) * (x - y)).sum(), ys)
                        };
                        let scale = |f: f32| e4m3_value(e4m3(amax / 6.0 * f)).max(f32::MIN_POSITIVE);
                        let mut best = quant(scale(1.0));
                        if search && amax > 0.0 {
                            for step in 0..=24 {
                                let candidate = quant(scale(0.70 + 0.025 * step as f32));
                                if candidate.0 < best.0 {
                                    best = candidate;
                                }
                            }
                        }
                        if amax == 0.0 {
                            best.1 = vec![0.0; 16];
                        }
                        for (i, y) in best.1.iter().enumerate() {
                            let at = ((row - first) * k + g * 16 + i) * 2;
                            chunk[at..at + 2].copy_from_slice(&((y.to_bits() >> 16) as u16).to_le_bytes());
                        }
                    }
                }
            });
        }
    });
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum KdaFp8 {
    /// BF16 KDA projections (the checkpoint's own precision).
    Off,
    /// One scale per output row.
    Channel,
    /// One scale per output row and 128-wide K block.
    Row128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layout {
    Block,
    Row128,
    Channel,
}

fn bf16(bytes: &[u8], i: usize) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])) << 16)
}

/// Quantizes a BF16 `[n, k]` weight; returns E4M3 bytes and FP32 scales in
/// `layout`, each block's scale by `rule`.
pub(crate) fn quantize(weight: &[u8], n: usize, k: usize, layout: Layout, rule: Fp8Scales) -> (Vec<u8>, Vec<f32>) {
    crate::shared::memory::load_timing::quantize(weight.len(), || quantize_inner(weight, n, k, layout, rule))
}

fn quantize_inner(weight: &[u8], n: usize, k: usize, layout: Layout, rule: Fp8Scales) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(weight.len(), n * k * 2);
    assert!(k % 128 == 0 && (layout != Layout::Block || n % 128 == 0));
    let kb = k / 128;
    let (scale_rows, rows_per_scale) = match layout {
        Layout::Block => (n / 128, 128),
        Layout::Row128 | Layout::Channel => (n, 1),
    };
    let mut values = vec![0u8; n * k];
    let mut scales = vec![0f32; scale_rows * kb];
    let threads = std::thread::available_parallelism().map_or(8, |p| p.get()).min(32);
    let per = scale_rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for ((values, scales), first) in values.chunks_mut(per * rows_per_scale * k)
            .zip(scales.chunks_mut(per * kb)).zip((0..scale_rows).step_by(per)) {
            scope.spawn(move || {
                for (local, srow) in (first..(first + per).min(scale_rows)).enumerate() {
                    let rows = srow * rows_per_scale..(srow + 1) * rows_per_scale;
                    // Scales for this scale row.
                    let mut s = vec![0f32; kb];
                    for row in rows.clone() {
                        for (b, sb) in s.iter_mut().enumerate() {
                            for c in b * 128..(b + 1) * 128 {
                                *sb = sb.max(bf16(weight, row * k + c).abs());
                            }
                        }
                    }
                    if layout == Layout::Channel {
                        let amax = s.iter().copied().fold(0f32, f32::max);
                        s.fill(amax);
                    }
                    let amax = s.clone();
                    for sb in &mut s {
                        *sb = match rule {
                            Fp8Scales::Pow2 => pow2_scale(*sb),
                            _ if *sb > 0.0 => *sb / 448.0,
                            _ => 1.0,
                        };
                    }
                    if rule == Fp8Scales::Best {
                        // Per scale block (all its rows), the scale with the smaller squared error.
                        let err = |b: usize, scale: f32| -> f64 {
                            rows.clone().map(|row| (b * 128..(b + 1) * 128).map(|c| {
                                let x = bf16(weight, row * k + c);
                                f64::from(e4m3_value(e4m3(x / scale)) * scale - x).powi(2)
                            }).sum::<f64>()).sum()
                        };
                        let blocks = if layout == Layout::Channel { 1 } else { kb };
                        for b in 0..blocks {
                            let (sa, sp) = (s[b], pow2_scale(amax[b]));
                            if sp != sa && err(b, sp) < err(b, sa) {
                                if layout == Layout::Channel {
                                    s.fill(sp);
                                } else {
                                    s[b] = sp;
                                }
                            }
                        }
                    }
                    for row in rows {
                        let out = &mut values[(row - first * rows_per_scale) * k..][..k];
                        for (c, q) in out.iter_mut().enumerate() {
                            *q = e4m3(bf16(weight, row * k + c) / s[c / 128]);
                        }
                    }
                    scales[local * kb..(local + 1) * kb].copy_from_slice(&s);
                }
            });
        }
    });
    (values, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(q: u8) -> f32 {
        e4m3_value(q)
    }

    #[test]
    fn e4m3_round_trips_and_rounds_to_nearest() {
        for q in 0u8..=0xFE {
            if q & 0x7F == 0x7F {
                continue;
            }
            assert_eq!(e4m3(decode(q)), q, "{q:#x} {}", decode(q));
        }
        assert_eq!(e4m3(500.0), 0x7E);
        assert_eq!(e4m3(450.0), 0x7E);
        // Halfway between 1.0 (0x38) and 1.125 (0x39) rounds to even (1.0).
        assert_eq!(e4m3(1.0625), 0x38);
        assert_eq!(e4m3(1.0626), 0x39);
        assert_eq!(e4m3(-0.0), 0x80);
    }

    #[test]
    fn pow2_scales_keep_three_mantissa_bits_exact() {
        assert_eq!(pow2_scale(448.0), 1.0);
        assert_eq!(pow2_scale(449.0), 2.0);
        assert_eq!(pow2_scale(1.75 * 2f32.powi(-3)), 2f32.powi(-11));
        assert_eq!(pow2_scale(0.0), 1.0);
        // A 128-wide row of 3-mantissa-bit values: exact under pow2 and best, not under amax / 448.
        let w: Vec<f32> = (0..128).map(|i| [1.0, 1.125, -1.5, 0.875, 3.25][i % 5] * 2f32.powi(-(i as i32 % 9))).collect();
        let bytes: Vec<u8> = w.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect();
        let exact = |rule| {
            let (q, s) = quantize(&bytes, 1, 128, Layout::Row128, rule);
            (0..128).filter(|&i| decode(q[i]) * s[0] == w[i]).count()
        };
        assert_eq!(exact(Fp8Scales::Pow2), 128);
        assert_eq!(exact(Fp8Scales::Best), 128);
        assert!(exact(Fp8Scales::Amax) < 128);
    }

    #[test]
    fn row_layouts_bound_the_error() {
        let (n, k) = (4, 256);
        let w: Vec<f32> = (0..n * k).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01 * (1 + i / k) as f32).collect();
        let bytes: Vec<u8> = w.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect();
        for layout in [Layout::Row128, Layout::Channel] {
            let (q, s) = quantize(&bytes, n, k, layout, Fp8Scales::Amax);
            for i in 0..n * k {
                let x = bf16(&bytes, i);
                let y = decode(q[i]) * s[(i / k) * 2 + (i % k) / 128];
                assert!((x - y).abs() <= x.abs() * 0.0625 + 1e-6, "{layout:?} {i}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn row128_head_slices_keep_whole_weight_quantization() {
        use crate::shared::peer_split::{slice_2d, Axis};

        // Non-block-aligned N, as in the packed split KDA input, and distinct
        // scales on either side of a 128-K boundary.
        let (n, k) = (34, 512);
        let bytes: Vec<u8> = (0..n * k).flat_map(|i| {
            let x = ((i * 37 % 101) as f32 - 50.0) * (1 + i / 128) as f32 * 0.01;
            ((x.to_bits() >> 16) as u16).to_le_bytes()
        }).collect();
        for rule in [Fp8Scales::Amax, Fp8Scales::Pow2, Fp8Scales::Best] {
            let (whole, scales) = quantize(&bytes, n, k, Layout::Row128, rule);
            let scale_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
            for axis in [Axis::Rows, Axis::Cols] {
                for rank in 0..2 {
                    let (rn, rk) = match axis { Axis::Rows => (n / 2, k), Axis::Cols => (n, k / 2) };
                    let part = slice_2d(&bytes, n, k, 2, axis, rank, 2);
                    let (q, s) = quantize(&part, rn, rk, Layout::Row128, rule);
                    assert_eq!(q, slice_2d(&whole, n, k, 1, axis, rank, 2));
                    let actual: Vec<u8> = s.iter().flat_map(|s| s.to_le_bytes()).collect();
                    assert_eq!(actual, slice_2d(&scale_bytes, n, k / 128, 4, axis, rank, 2));
                }
            }
        }
    }
    /// Inputs that reach every branch of the host rule: zeros, subnormal and
    /// saturating quotients, a block whose pow2 and amax/448 errors nearly tie,
    /// exact E4M3 values, and a spread of magnitudes.
    fn load_quant_cases() -> Vec<(Layout, Fp8Scales, usize, usize, Vec<u8>)> {
        let mut cases = Vec::new();
        let bf = |x: f32| ((x.to_bits() >> 16) as u16).to_le_bytes();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        for (layout, rows) in [(Layout::Block, 256), (Layout::Row128, 6), (Layout::Channel, 6)] {
            for rule in [Fp8Scales::Amax, Fp8Scales::Pow2, Fp8Scales::Best] {
                let cols = 384;
                let mut bytes = Vec::with_capacity(rows * cols * 2);
                for r in 0..rows {
                    for c in 0..cols {
                        let block = (r / if layout == Layout::Block { 128 } else { 1 }, c / 128);
                        let x = match (block.0 + block.1) % 6 {
                            0 => 0.0,
                            1 => ((next() % 2001) as f32 - 1000.0) * 1e-6,
                            2 => ((next() % 2001) as f32 - 1000.0) * 3.7,
                            // Values on the E4M3 grid times 2^-3: pow2 quantizes exactly.
                            3 => e4m3_value((next() % 0x7E) as u8 | (((next() & 1) as u8) << 7)) * 0.125,
                            4 => f32::from_bits(((next() % 30 + 110) as u32) << 23 | (next() as u32 & 0x7F_0000))
                                * if next() & 1 == 1 { -1.0 } else { 1.0 },
                            _ => ((next() % 65_536) as f32 - 32_768.0) / 1024.0,
                        };
                        bytes.extend(bf(x));
                    }
                }
                cases.push((layout, rule, rows, cols, bytes));
            }
        }
        cases
    }

    fn load_quant_fixture() -> Vec<u8> {
        let cases = load_quant_cases();
        let mut out = b"LQF1".to_vec();
        out.extend((cases.len() as u32).to_le_bytes());
        for (layout, rule, rows, cols, bytes) in cases {
            let (q, scales) = quantize(&bytes, rows, cols, layout, rule);
            let layout = match layout { Layout::Block => 0i32, Layout::Row128 => 1, Layout::Channel => 2 };
            out.extend(layout.to_le_bytes());
            out.extend(rule.code().to_le_bytes());
            out.extend((rows as u32).to_le_bytes());
            out.extend((cols as u32).to_le_bytes());
            out.extend(&bytes);
            out.extend(&q);
            out.extend(scales.iter().flat_map(|s| s.to_le_bytes()));
        }
        out
    }

    /// The native load kernels' CPU self-test (native/tests/fp8_load_quant_selftest.cc)
    /// compares against this fixture; it must equal today's host rule byte for byte.
    /// Regenerate with CUTEAFD_WRITE_LOAD_QUANT_FIXTURE=1.
    #[test]
    fn native_load_quant_fixture_matches_host_rule() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../native/tests/fixtures/fp8_load_quant.bin");
        let fixture = load_quant_fixture();
        if std::env::var_os("CUTEAFD_WRITE_LOAD_QUANT_FIXTURE").is_some() {
            std::fs::write(&path, &fixture).unwrap();
        }
        assert!(std::fs::read(&path).unwrap() == fixture, "native load-quant fixture differs from the host rule");
    }
}

