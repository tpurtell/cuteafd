//! Ignored real-checkpoint equality gates for the shared rank backends.
use super::*;
use crate::shared::experts::rtx::{
    exl3::Exl3Tp2, native::NativeTp2, ExpertInput, Routes, RtxExpertLayer,
};
use cuteafd_ffi::NativeLibrary;

fn host(device: Device<'_>, buffer: CuteafdDeviceBuffer, bytes: usize) -> Result<Vec<u8>> {
    let mut data = vec![0; bytes];
    device.run(|| {
        device
            .library
            .copy_d2h(&mut data, CuteafdDeviceBuffer { bytes, ..buffer })
    })?;
    Ok(data)
}
fn errors(a: &[u8], b: &[u8]) -> (f64, f64) {
    let mut abs = 0f64;
    let mut square = 0f64;
    let mut signal = 0f64;
    for (a, b) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let x = f32::from_ne_bytes(a.try_into().unwrap()) as f64;
        let y = f32::from_ne_bytes(b.try_into().unwrap()) as f64;
        assert!(x.is_finite() && y.is_finite());
        abs = abs.max((x - y).abs());
        square += (x - y).powi(2);
        signal += x * x;
    }
    (abs, (square / signal.max(f64::MIN_POSITIVE)).sqrt())
}
fn fp32_sum(ranks: &[Vec<u8>; 2]) -> Vec<f32> {
    ranks[0]
        .chunks_exact(4)
        .zip(ranks[1].chunks_exact(4))
        .map(|(a, b)| {
            f32::from_ne_bytes(a.try_into().unwrap()) + f32::from_ne_bytes(b.try_into().unwrap())
        })
        .collect()
}
fn rounded_bf16(sum: &[f32]) -> Vec<u8> {
    sum.iter()
        .flat_map(|v| {
            let bits = v.to_bits();
            ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16).to_ne_bytes()
        })
        .collect()
}
fn bf16_value(bits: u16) -> f64 {
    f32::from_bits((bits as u32) << 16) as f64
}

// Round interval endpoints directly from FP64, avoiding an intermediate FP32
// rounding that could move an endpoint onto an RN-even BF16 tie.
fn rn_bf16(value: f64) -> f64 {
    let magnitude = value.abs();
    let mut lower_bits = ((magnitude as f32).to_bits() >> 16) as u16;
    if bf16_value(lower_bits) > magnitude {
        lower_bits -= 1;
    }
    let lower = bf16_value(lower_bits);
    let upper = bf16_value(lower_bits + 1);
    let rounded = match (magnitude - lower).total_cmp(&(upper - magnitude)) {
        std::cmp::Ordering::Less => lower,
        std::cmp::Ordering::Greater => upper,
        std::cmp::Ordering::Equal if lower_bits & 1 == 0 => lower,
        std::cmp::Ordering::Equal => upper,
    };
    rounded.copysign(value)
}

struct FlipBounds {
    interval: bool,
    single_spacing: bool,
    boundary: bool,
    near_zero: bool,
    lo: f64,
    hi: f64,
    spacing: f64,
}

fn flip_bounds(x: f32, y: f32, rounded_x: f64, rounded_y: f64, delta: f64) -> FlipBounds {
    let lo = rn_bf16(x as f64 - delta);
    let hi = rn_bf16(x as f64 + delta);
    let magnitude = x.abs().max(y.abs());
    let floor_bits = (magnitude.to_bits() >> 16) as u16;
    let spacing = bf16_value(floor_bits + 1) - bf16_value(floor_bits);
    let near_zero = magnitude as f64 <= delta * 1024.0;
    let bits = ((rounded_x.abs() as f32).to_bits() >> 16) as u16;
    let upward = rounded_y > rounded_x;
    let neighbor = if rounded_x == 0.0 {
        bf16_value(1).copysign(if upward { 1.0 } else { -1.0 })
    } else if upward != rounded_x.is_sign_negative() {
        bf16_value(bits + 1).copysign(rounded_x)
    } else {
        bf16_value(bits - 1).copysign(rounded_x)
    };
    let boundary = (rounded_x + neighbor) / 2.0;
    FlipBounds {
        interval: lo <= rounded_y && rounded_y <= hi,
        single_spacing: near_zero || (rounded_x - rounded_y).abs() <= spacing,
        boundary: (x as f64).min(y as f64) <= boundary && boundary <= (x as f64).max(y as f64),
        near_zero,
        lo,
        hi,
        spacing,
    }
}

#[derive(Default)]
struct AtomicMetrics {
    flips: usize,
}
fn atomic_flip_ceiling(self_counts: &[usize]) -> usize {
    self_counts.iter().copied().max().unwrap_or(0) * 2
}
fn atomic_flip_within_ceiling(shared: usize, self_counts: &[usize]) -> bool {
    shared <= atomic_flip_ceiling(self_counts)
}

#[test]
fn atomic_bf16_flip_sanity_covers_both_measured_count_runs() {
    for (shared, counts) in [
        (100, [106, 119, 108]),
        (91, [74, 77, 94]),
        (112, [97, 70, 79]),
        (110, [98, 115, 91]),
    ] {
        assert!(atomic_flip_within_ceiling(shared, &counts));
        assert!(!atomic_flip_within_ceiling(
            atomic_flip_ceiling(&counts) + 1,
            &counts
        ));
    }
    assert_eq!(atomic_flip_ceiling(&[0, 0, 0]), 0);
}
fn atomic_owner_metrics(
    label: &str,
    rows: usize,
    a: &[f32],
    b: &[f32],
    aa: &[u8],
    bb: &[u8],
) -> Result<AtomicMetrics> {
    let bytes = |sum: &[f32]| sum.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>();
    let (max_abs, rel_l2) = errors(&bytes(a), &bytes(b));
    let mut flips = 0;
    let mut max_steps = 0;
    let mut max_position = 0;
    let mut away = 0;
    let mut multiple = 0;
    let mut outside_interval = 0;
    let mut outside_spacing = 0;
    let mut bf16_max_abs = 0f64;
    for (position, ((x, y), (u, v))) in a
        .iter()
        .zip(b)
        .zip(aa.chunks_exact(2).zip(bb.chunks_exact(2)))
        .enumerate()
    {
        let u = u16::from_ne_bytes(u.try_into().unwrap());
        let v = u16::from_ne_bytes(v.try_into().unwrap());
        if u == v {
            continue;
        }
        flips += 1;
        let steps = u.abs_diff(v);
        if steps > max_steps {
            max_steps = steps;
            max_position = position;
        }
        multiple += usize::from(steps > 1);
        let rounded_x = bf16_value(u);
        let rounded_y = bf16_value(v);
        let bf16_delta = (rounded_x - rounded_y).abs();
        bf16_max_abs = bf16_max_abs.max(bf16_delta);
        let bounds = flip_bounds(*x, *y, rounded_x, rounded_y, max_abs);
        eprintln!("V41_ATOMIC_FLIP label={label} rows={rows} position={position} x={x:.9e} y={y:.9e} delta={:.9e} steps={steps} bf16_abs={bf16_delta:.9e} lo={:.9e} hi={:.9e} spacing={:.9e} near_zero={}", (*x as f64 - *y as f64).abs(), bounds.lo, bounds.hi, bounds.spacing, bounds.near_zero);
        outside_interval += usize::from(!bounds.interval);
        outside_spacing += usize::from(!bounds.single_spacing);
        away += usize::from(!bounds.boundary);
    }
    eprintln!("V41_ATOMIC_OWNER label={label} rows={rows} pre_max_abs={max_abs:.9e} pre_rel_l2={rel_l2:.9e} flips={flips} max_steps={max_steps} max_position={max_position} multiple_steps={multiple} away_from_boundary={away} bf16_max_abs={bf16_max_abs:.9e} outside_interval={outside_interval} outside_spacing={outside_spacing}");
    ensure!(rel_l2 < 1e-6, "atomic pre-round owner tolerance");
    ensure!(
        outside_interval == 0 && outside_spacing == 0 && away == 0,
        "atomic BF16 flips outside reachable rounding interval/spacing"
    );
    Ok(AtomicMetrics { flips })
}

#[test]
fn atomic_bf16_interval_endpoint_rounding_is_exact() {
    for bits in [0, 1, 2, 0x0080, 0x36f8, 0x3f00, 0x3f01, 0x3f80, 0x7e00] {
        let lo = bf16_value(bits);
        let hi = bf16_value(bits + 1);
        let tie = (lo + hi) / 2.0;
        for sign in [1.0, -1.0] {
            assert_eq!(rn_bf16(sign * lo), sign * lo);
            assert_eq!(rn_bf16(sign * hi), sign * hi);
            assert_eq!(
                rn_bf16(sign * tie),
                sign * if bits & 1 == 0 { lo } else { hi }
            );
        }
    }
}

#[test]
fn atomic_bf16_interval_preserves_near_zero_even_ties() {
    let x = 7.405877113e-6_f32;
    let y = 7.435679436e-6_f32;
    let delta = 9.5367431640625e-7;
    for position in [28268, 355948, 683628, 1011308] {
        for sign in [1.0, -1.0] {
            let x = x * sign;
            let y = y * sign;
            let a = rn_bf16(x as f64);
            let b = rn_bf16(y as f64);
            let bounds = flip_bounds(x, y, a, b, delta);
            assert!(
                bounds.near_zero && bounds.interval && bounds.boundary,
                "{position}"
            );
            assert!((a - b).abs() > bounds.spacing);
            assert!(bounds.single_spacing);
            assert!(!flip_bounds(x, y, a, bounds.hi + bounds.spacing, delta).interval);
        }
    }
}

#[test]
fn atomic_bf16_spacing_is_local_to_each_flip_magnitude() {
    let delta = 9.5367431640625e-7;
    let mut differences = Vec::new();
    for x in [0.501953125_f32, 0.2509765625_f32] {
        let y = f32::from_bits(x.to_bits() + 1);
        let a = rn_bf16(x as f64);
        let b = rn_bf16(y as f64);
        let bounds = flip_bounds(x, y, a, b, delta);
        assert!(!bounds.near_zero);
        assert!(bounds.interval && bounds.single_spacing && bounds.boundary);
        assert_eq!((a - b).abs(), bounds.spacing);
        differences.push((a - b).abs());
    }
    assert_eq!(differences, [0.00390625, 0.001953125]);
}

fn inputs<'a>(
    devices: [Device<'a>; 2],
    capacity: usize,
) -> Result<Vec<([Allocation<'a>; 3], Stream<'a>)>> {
    let geometry = cuteafd_core::expert_geometry();
    let h = geometry.hidden as usize;
    let k = geometry.topk as usize;
    let mut wire = vec![0; capacity * (h + h / 32)];
    for (row, bytes) in wire.chunks_exact_mut(h + h / 32).enumerate() {
        for (column, value) in bytes[..h].iter_mut().enumerate() {
            *value = [0x30, 0xb0, 0x38, 0xb8][(column + row) % 4];
        }
        bytes[h..].fill(127);
    }
    let ids: Vec<_> = (0..capacity * k)
        .flat_map(|i| ((i * 7 % geometry.experts as usize) as u32).to_ne_bytes())
        .collect();
    let routing: Vec<_> = (0..capacity * k)
        .flat_map(|i| ((i % k + 1) as f32 / (k * (k + 1) / 2) as f32).to_ne_bytes())
        .collect();
    devices
        .into_iter()
        .map(|device| {
            let buffers = [
                Allocation::new(device, wire.len())?,
                Allocation::new(device, ids.len())?,
                Allocation::new(device, routing.len())?,
            ];
            device.run(|| {
                for (buffer, data) in buffers.iter().zip([&wire, &ids, &routing]) {
                    device.library.copy_h2d(buffer.buffer, data)?;
                }
                Ok(())
            })?;
            Ok((buffers, Stream::new(device)?))
        })
        .collect()
}

#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT and two GPUs"]
fn shared_native_v41_byte_exact() -> Result<()> {
    v41_equality(false)
}

#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT, CUTEAFD_EXL3_AOT and two GPUs"]
fn shared_exl3_v41_byte_exact() -> Result<()> {
    v41_equality(true)
}

fn v41_equality(exl3: bool) -> Result<()> {
    // SAFETY: operator-selected trusted native library, retained through all owners.
    let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    lib.cuda_set_device(0)?;
    let devices = [
        Device {
            library: &lib,
            id: 0,
        },
        Device {
            library: &lib,
            id: 1,
        },
    ];
    let catalog = cuteafd_loader::read_official_v41_catalog(
        cuteafd_loader::OFFICIAL_V41_MODEL_ID,
        Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?),
    )?;
    let capacity = 256;
    let input = inputs(devices, capacity)?;
    let package = PathBuf::from(std::env::var("CUTEAFD_EXL3_AOT").unwrap_or_default());
    let weights = if exl3 {
        RankWeights::load_exl3_pair(devices, &catalog, 1, [8_000_000_000; 2], &package)?
    } else {
        [
            RankWeights::load(devices[0], &catalog, 1, 8_000_000_000)?,
            RankWeights::load(devices[1], &catalog, 1, 8_000_000_000)?,
        ]
    }
    .map(Rc::new);
    let mut waves = [
        RankWave::new(weights[0].clone(), capacity as u32)?,
        RankWave::new(weights[1].clone(), capacity as u32)?,
    ];
    let mut full_wave = ExpertWave::new(weights, capacity as u32)?;
    let mut shared: [Box<dyn RtxExpertLayer + '_>; 2] = if exl3 {
        let [a, b] = Exl3Tp2::load_pair(
            devices,
            &catalog,
            &package,
            0..1,
            capacity,
            [8_000_000_000; 2],
        )?;
        [Box::new(a), Box::new(b)]
    } else {
        [
            Box::new(NativeTp2::load(
                devices[0],
                &catalog,
                0,
                0..1,
                capacity,
                8_000_000_000,
            )?),
            Box::new(NativeTp2::load(
                devices[1],
                &catalog,
                1,
                0..1,
                capacity,
                8_000_000_000,
            )?),
        ]
    };
    let copied = [
        Allocation::new(devices[0], capacity * 5120 * 4)?,
        Allocation::new(devices[1], capacity * 5120 * 4)?,
    ];
    let mut reduction = PeerReduction::new(devices[1], devices[0], capacity as u32)?;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    let mut failed_atomic = Vec::new();
    for rows in [1, 3, 16, 80, 255, 256] {
        let mut shared_values = [Vec::new(), Vec::new()];
        for rank in 0..2 {
            let (buffers, stream) = &input[rank];
            // SAFETY: the fixture owns all inputs/outputs and drains each producer before inspection.
            let layout = unsafe {
                waves[rank].enqueue(
                    0,
                    rows as u32,
                    buffers[0].buffer,
                    buffers[1].buffer,
                    buffers[2].buffer,
                    stream,
                )?
            };
            waves[rank].stream.drain()?;
            ensure!(
                layout == Tp2RoutedLayout::Fp32Tokens,
                "set CUTEAFD_TP2_TOKEN_SUMS=1 for rank equality"
            );
            // SAFETY: identical rank-local inputs, exclusive fixture stream and workspace.
            unsafe {
                shared[rank].enqueue(
                    0,
                    rows,
                    ExpertInput::Fp8K32(buffers[0].buffer.ptr),
                    Routes {
                        ids: buffers[1].buffer.ptr,
                        weights: buffers[2].buffer.ptr,
                    },
                    stream.raw,
                )?;
            }
            stream.drain()?;
            let reference = host(devices[rank], waves[rank].output.buffer, rows * 5120 * 4)?;
            let source = CuteafdDeviceBuffer {
                ptr: shared[rank].output(),
                bytes: rows * 5120 * 4,
                device_id: rank as i32,
                flags: 0,
            };
            let actual = host(devices[rank], source, source.bytes)?;
            shared_values[rank] = actual.clone();
            let (max_abs, rel_l2) = errors(&reference, &actual);
            eprintln!("V41_SHARED backend={} rows={rows} rank={rank} identical={} max_abs={max_abs:.9e} rel_l2={rel_l2:.9e}",
                if exl3 {"exl3"} else {"native"}, reference==actual);
            if rows <= 80 || exl3 {
                ensure!(reference == actual, "V4.1 shared FP32 rank mismatch");
            } else {
                ensure!(rel_l2 < 1e-6, "V4.1 atomic rank tolerance exceeded");
            }
            // SAFETY: inspected source is complete and the staging owner outlives reduction.
            devices[rank].run(|| unsafe {
                lib.copy_d2d_async(copied[rank].buffer, source, source.bytes, stream.raw)
            })?;
            stream.drain()?;
        }
        // SAFETY: staging and streams live until the cooperative reduction completes.
        let output = runtime.block_on(unsafe {
            reduction.reduce(
                &copied[0],
                &copied[1],
                &input[0].1,
                &input[1].1,
                rows as u32,
                Tp2RoutedLayout::Fp32Tokens,
            )
        })?;
        let actual = host(devices[0], output, rows * 5120 * 2)?;
        let make_inputs = || {
            std::array::from_fn(|rank| RankInputs {
                wire: input[rank].0[0].buffer,
                ids: input[rank].0[1].buffer,
                routing: input[rank].0[2].buffer,
                producer: &input[rank].1,
            })
        };
        // SAFETY: the fixture retains identical inputs through this independent expert wave.
        let reference =
            runtime.block_on(unsafe { full_wave.execute(0, rows as u32, 0, make_inputs()) })?;
        let expected = host(devices[0], reference, rows * 5120 * 2)?;
        let identical = expected == actual;
        eprintln!(
            "V41_SHARED_REDUCE backend={} rows={rows} identical={identical}",
            if exl3 { "exl3" } else { "native" }
        );
        if rows <= 80 || exl3 {
            ensure!(identical, "V4.1 owner reduction mismatch");
        } else {
            let production_values = [
                host(
                    devices[0],
                    full_wave.ranks[0].output.buffer,
                    rows * 5120 * 4,
                )?,
                host(
                    devices[1],
                    full_wave.ranks[1].output.buffer,
                    rows * 5120 * 4,
                )?,
            ];
            let sum_shared = fp32_sum(&shared_values);
            let sum_production = fp32_sum(&production_values);
            ensure!(
                rounded_bf16(&sum_shared) == actual,
                "shared owner differs from its FP32 reference then round"
            );
            ensure!(
                rounded_bf16(&sum_production) == expected,
                "production owner differs from FP32 reference then round"
            );
            let shared_check = atomic_owner_metrics(
                "shared-vs-production",
                rows,
                &sum_shared,
                &sum_production,
                &actual,
                &expected,
            );
            let mut self_envelope = AtomicMetrics::default();
            let mut previous_sum = sum_production;
            let mut previous_bf16 = expected;
            let mut self_valid = true;
            let mut self_counts = Vec::new();
            // Atomic boundary flip counts are stochastic; per-element checks are
            // hard gates, and three self transitions supply a loose sanity ceiling.
            for repeat in 0..3 {
                // SAFETY: retain identical input owners through each production run.
                let next = runtime
                    .block_on(unsafe { full_wave.execute(0, rows as u32, 0, make_inputs()) })?;
                let next_bf16 = host(devices[0], next, rows * 5120 * 2)?;
                let next_values = [
                    host(
                        devices[0],
                        full_wave.ranks[0].output.buffer,
                        rows * 5120 * 4,
                    )?,
                    host(
                        devices[1],
                        full_wave.ranks[1].output.buffer,
                        rows * 5120 * 4,
                    )?,
                ];
                let next_sum = fp32_sum(&next_values);
                ensure!(
                    rounded_bf16(&next_sum) == next_bf16,
                    "production owner differs from FP32 reference then round"
                );
                match atomic_owner_metrics(
                    &format!("production-vs-production-{repeat}"),
                    rows,
                    &previous_sum,
                    &next_sum,
                    &previous_bf16,
                    &next_bf16,
                ) {
                    Ok(metrics) => {
                        self_envelope.flips = self_envelope.flips.max(metrics.flips);
                        self_counts.push(metrics.flips);
                    }
                    Err(_) => self_valid = false,
                }
                previous_sum = next_sum;
                previous_bf16 = next_bf16;
            }
            let ceiling = atomic_flip_ceiling(&self_counts);
            let within_envelope = shared_check
                .is_ok_and(|metrics| atomic_flip_within_ceiling(metrics.flips, &self_counts));
            eprintln!("V41_ATOMIC_ENVELOPE rows={rows} self_counts={self_counts:?} self_max_flips={} sanity_ceiling={ceiling} shared_within={within_envelope}", self_envelope.flips);
            if !within_envelope || !self_valid {
                failed_atomic.push(rows);
            }
        }
        ensure!(
            lib.cuda_get_device()? == 0,
            "shared backend leaked current device"
        );
    }
    ensure!(
        failed_atomic.is_empty(),
        "atomic owner tolerance exceeded: {failed_atomic:?}"
    );
    Ok(())
}
