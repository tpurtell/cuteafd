//! Compare shared rank partials against LocalExperts before its BF16 finish.
use super::*;
use crate::shared::experts::rtx::{
    exl3::Exl3Tp2, native::NativeTp2, ExpertInput, Routes, RtxExpertLayer,
};
use crate::shared::memory::device::{Allocation, Device, Stream};

#[test]
#[ignore = "requires Flash snapshot, C1 native library and two GPUs"]
fn shared_native_flash_matches_tp1() -> Result<()> {
    compare(false)
}
#[test]
#[ignore = "requires Pro EXL3 snapshot, CUTEAFD_EXL3_AOT package and two GPUs"]
fn shared_exl3_pro_matches_tp1() -> Result<()> {
    compare(true)
}

fn compare(exl3: bool) -> Result<()> {
    let geometry = if exl3 {
        cuteafd_core::ExpertGeometry::DEEPSEEK_V4_PRO
    } else {
        cuteafd_core::ExpertGeometry::DEEPSEEK_V4_FLASH
    };
    cuteafd_core::set_expert_geometry(geometry)
        .map_err(|g| anyhow::anyhow!("fixture geometry already {g:?}"))?;
    let native = std::path::PathBuf::from(std::env::var("CUTEAFD_NATIVE_LIB")?);
    // SAFETY: trusted operator-provided library, retained through every owner.
    let lib = unsafe { NativeLibrary::load(&native)? };
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
    let catalog =
        cuteafd_loader::read_expert_catalog(Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
    let h = geometry.hidden as usize;
    let k = geometry.topk as usize;
    let capacity = if exl3 { 80 } else { 1024 };
    let mut wire = vec![0; capacity * (h + h / 32)];
    for (r, row) in wire.chunks_exact_mut(h + h / 32).enumerate() {
        for (c, x) in row[..h].iter_mut().enumerate() {
            *x = [0x30, 0xb0, 0x38, 0xb8][(r + c) % 4];
        }
        row[h..].fill(127);
    }
    let ids: Vec<_> = (0..capacity * k)
        .flat_map(|i| ((i * 7 % geometry.experts as usize) as u32).to_ne_bytes())
        .collect();
    let weights: Vec<_> = (0..capacity * k)
        .flat_map(|i| ((i % k + 1) as f32 / (k * (k + 1) / 2) as f32).to_ne_bytes())
        .collect();
    let inputs = devices
        .into_iter()
        .map(|device| {
            let buffers = [
                Allocation::new(device, wire.len())?,
                Allocation::new(device, ids.len())?,
                Allocation::new(device, weights.len())?,
            ];
            device.run(|| {
                for (b, data) in buffers.iter().zip([&wire, &ids, &weights]) {
                    lib.copy_h2d(b.buffer, data)?;
                }
                Ok(())
            })?;
            Ok((buffers, Stream::new(device)?))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut local = devices[0].own(|| {
        LocalExperts::load_range(
            &lib,
            &native,
            &catalog,
            0,
            0..1,
            capacity,
            16_000_000_000,
            inputs[0].1.raw,
        )?
        .context("TP1 reference not built")
    })?;
    let mut shared: [Box<dyn RtxExpertLayer + '_>; 2] = if exl3 {
        let [a, b] = Exl3Tp2::load_pair(
            devices,
            &catalog,
            Path::new(&std::env::var("CUTEAFD_EXL3_AOT")?),
            0..1,
            capacity,
            [10_000_000_000; 2],
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
                10_000_000_000,
            )?),
            Box::new(NativeTp2::load(
                devices[1],
                &catalog,
                1,
                0..1,
                capacity,
                10_000_000_000,
            )?),
        ]
    };
    let read = |device: Device<'_>, buffer: CuteafdDeviceBuffer| -> Result<Vec<f32>> {
        let mut data = vec![0; buffer.bytes];
        device.run(|| lib.copy_d2h(&mut data, buffer))?;
        Ok(data
            .chunks_exact(4)
            .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
            .collect())
    };
    let tp1_sums = Allocation::new(devices[0], capacity * h * 4)?;
    let mut failures = Vec::new();
    for rows in if exl3 {
        vec![1, 16, 80]
    } else {
        vec![1, 16, 80, 256, 1024]
    } {
        // SAFETY: fixture owns all inputs, no shared addition and the stream drains before reads.
        devices[0].run(|| unsafe {
            local.run(
                LocalLayer::Backbone(0),
                rows,
                inputs[0].0[0].buffer.ptr,
                inputs[0].0[1].buffer.ptr,
                inputs[0].0[2].buffer.ptr,
                std::ptr::null_mut(),
                inputs[0].1.raw,
            )
        })?;
        inputs[0].1.drain()?;
        let mut reference = devices[0].run(|| {
            match &mut local.backend {
                Backend::Exl3 { executions, .. } => {
                    let execution = executions
                        .iter_mut()
                        .find(|e| e.capacity() >= rows)
                        .unwrap();
                    // SAFETY: complete rank-local fixture inputs, no overlapping scratch use.
                    unsafe {
                        execution.launch_layer(
                            0,
                            [
                                inputs[0].0[0].buffer,
                                inputs[0].0[1].buffer,
                                inputs[0].0[2].buffer,
                            ],
                            rows,
                            inputs[0].1.raw,
                        )
                    }
                }
                Backend::Native { states, .. } => {
                    let state = states
                        .iter()
                        .find(|s| s.kernel.info().capacity_rows as usize >= rows)
                        .unwrap();
                    let source = CuteafdDeviceBuffer {
                        ptr: state.slots[41],
                        bytes: rows * h * 4,
                        device_id: 0,
                        flags: 0,
                    };
                    if state.kernel.output_kind() == cuteafd_ffi::V41ExpertOutputKind::Fp32Routes {
                        // SAFETY: complete TP1 route planes, disjoint fixture-owned token output.
                        unsafe {
                            lib.v41_tp2_expert_reducer()?.sum_routes_geometry(
                                CuteafdDeviceBuffer {
                                    bytes: source.bytes * k,
                                    ..source
                                },
                                tp1_sums.buffer,
                                rows as u32,
                                geometry,
                                inputs[0].1.raw,
                            )?;
                        }
                        Ok(tp1_sums.buffer)
                    } else {
                        ensure!(
                            state.kernel.output_kind()
                                == cuteafd_ffi::V41ExpertOutputKind::Fp32Tokens,
                            "TP1 fixture needs FP32 output"
                        );
                        Ok(source)
                    }
                }
            }
        })?;
        inputs[0].1.drain()?;
        reference.bytes = rows * h * 4;
        let expected = read(devices[0], reference)?;
        let mut ranks = Vec::new();
        for rank in 0..2 {
            // SAFETY: identical owned input rows, exclusive rank-local stream and output.
            unsafe {
                shared[rank].enqueue(
                    0,
                    rows,
                    ExpertInput::Fp8K32(inputs[rank].0[0].buffer.ptr),
                    Routes {
                        ids: inputs[rank].0[1].buffer.ptr,
                        weights: inputs[rank].0[2].buffer.ptr,
                    },
                    inputs[rank].1.raw,
                )?;
            }
            inputs[rank].1.drain()?;
            ranks.push(read(
                devices[rank],
                CuteafdDeviceBuffer {
                    ptr: shared[rank].output(),
                    bytes: rows * h * 4,
                    device_id: rank as i32,
                    flags: 0,
                },
            )?);
        }
        let mut max_abs = 0f64;
        let mut error = 0f64;
        let mut signal = 0f64;
        for ((x, a), b) in expected.iter().zip(&ranks[0]).zip(&ranks[1]) {
            let sum = *a + *b;
            ensure!(x.is_finite() && sum.is_finite(), "nonfinite TP1/TP2 output");
            let delta = (*x as f64) - (sum as f64);
            max_abs = max_abs.max(delta.abs());
            error += delta * delta;
            signal += (*x as f64).powi(2);
        }
        let rel_l2 = (error / signal.max(f64::MIN_POSITIVE)).sqrt();
        eprintln!(
            "V4_TP2_VS_TP1 format={} rows={rows} max_abs={max_abs:.9e} rel_l2={rel_l2:.9e}",
            if exl3 { "pro-exl3" } else { "flash-native" }
        );
        if let Ok(directory) = std::env::var("CUTEAFD_TP2_FIXTURE_OUTPUT") {
            let prefix = Path::new(&directory)
                .join(format!("{}-{rows}", if exl3 { "pro" } else { "flash" }));
            let bytes = |values: &[f32]| {
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()
            };
            std::fs::write(prefix.with_extension("tp1.f32"), bytes(&expected))?;
            let summed: Vec<_> = ranks[0].iter().zip(&ranks[1]).map(|(a, b)| a + b).collect();
            std::fs::write(prefix.with_extension("tp2.f32"), bytes(&summed))?;
        }
        // EXL3 includes FP16 rotation stores. When given an independently decoded
        // FP64 oracle, TP2 must be no worse than twice TP1's distance to it.
        // Without that oracle retain the strict TP1 comparison, never silently skip.
        if exl3 && std::env::var_os("CUTEAFD_TP2_FP64_REFERENCE").is_some() {
            let reference = std::fs::read(std::env::var("CUTEAFD_TP2_FP64_REFERENCE")?)?;
            ensure!(
                reference.len() >= rows * h * 8,
                "short FP64 expert reference"
            );
            let mut tp1_error = 0f64;
            let mut tp2_error = 0f64;
            let mut signal = 0f64;
            let mut tp1_abs = 0f64;
            let mut tp2_abs = 0f64;
            for (i, chunk) in reference[..rows * h * 8].chunks_exact(8).enumerate() {
                let oracle = f64::from_le_bytes(chunk.try_into().unwrap());
                ensure!(oracle.is_finite(), "nonfinite FP64 oracle");
                let a = expected[i] as f64 - oracle;
                let b = (ranks[0][i] + ranks[1][i]) as f64 - oracle;
                tp1_error += a * a;
                tp2_error += b * b;
                signal += oracle * oracle;
                tp1_abs = tp1_abs.max(a.abs());
                tp2_abs = tp2_abs.max(b.abs());
            }
            let tp1_l2 = (tp1_error / signal.max(f64::MIN_POSITIVE)).sqrt();
            let tp2_l2 = (tp2_error / signal.max(f64::MIN_POSITIVE)).sqrt();
            eprintln!("V4_TP2_FP64 rows={rows} tp1_max_abs={tp1_abs:.9e} tp1_rel_l2={tp1_l2:.9e} tp2_max_abs={tp2_abs:.9e} tp2_rel_l2={tp2_l2:.9e}");
            if tp2_error > 4.0 * tp1_error {
                failures.push((rows, tp2_l2));
            }
        } else if rel_l2 >= if rows >= 256 { 1e-6 } else { 1e-4 } {
            failures.push((rows, rel_l2));
        }
    }
    ensure!(
        failures.is_empty(),
        "TP2 versus TP1 exceeds reassociation tolerance: {failures:?}"
    );
    Ok(())
}
