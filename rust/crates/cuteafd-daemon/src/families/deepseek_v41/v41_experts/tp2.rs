//! V4.1 TP2 qualification fixtures. Production execution lives in shared/experts/rtx.
#[cfg(test)]
use crate::shared::{
    experts::rtx::{
        combine::{Combine, OwnerReduce},
        exl3::Exl3Tp2,
        native::{NativeOutputPolicy, NativeTp2},
        pair::{OwnerInputs, OwnerPair, OwnerWeights},
        ExpertInput, PartialDtype, PartialLayout, Routes, RtxExpertLayer, RtxSelectionPolicy,
    },
    memory::device::{Allocation, Device, Stream},
};
#[cfg(test)]
use anyhow::{ensure, Result};
#[cfg(test)]
use cuteafd_ffi::CuteafdDeviceBuffer;
#[cfg(test)]
use std::path::{Path, PathBuf};

#[cfg(test)]
#[path = "tp2/shared_tests.rs"]
mod shared_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_ffi::NativeLibrary;

    #[test]
    #[ignore = "requires EXL3 snapshot, TP2 package and two CUDA devices"]
    fn exl3_single_rank_capacity_launch_probe() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let device = Device {
            library: &lib,
            id: 0,
        };
        let catalog = cuteafd_loader::read_official_v41_catalog(Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?),
        )?;
        let package = PathBuf::from(std::env::var("CUTEAFD_EXL3_AOT")?);
        let devices = [
            device,
            Device {
                library: &lib,
                id: 1,
            },
        ];
        let weights = Exl3Tp2::load_pair_weights(
            devices,
            &catalog,
            cuteafd_core::expert_geometry(),
            [&package; 2],
            vec![0],
            16,
            [4_000_000_000; 2],
            RtxSelectionPolicy::FirstCapacity,
        )?;
        let inputs = [
            Allocation::new(device, 16 * 5280)?,
            Allocation::new(device, 16 * 24)?,
            Allocation::new(device, 16 * 24)?,
        ];
        let producer = Stream::new(device)?;
        let mut wave = weights[0].execution()?;
        let mut wire = vec![0u8; 16 * 5280];
        for row in wire.chunks_exact_mut(5280) {
            row[..5120].fill(0x38);
            row[5120..].fill(127);
        }
        lib.copy_h2d(inputs[0].buffer, &wire)?;
        lib.copy_h2d(
            inputs[1].buffer,
            &(0..96i32)
                .flat_map(|i| (i % 6).to_ne_bytes())
                .collect::<Vec<_>>(),
        )?;
        lib.copy_h2d(
            inputs[2].buffer,
            &(0..96)
                .flat_map(|_| (1f32 / 6.).to_ne_bytes())
                .collect::<Vec<_>>(),
        )?;
        for rows in [16, 1, 3, 16] {
            unsafe {
                wave.enqueue(
                    0,
                    rows,
                    ExpertInput::Fp8K32(inputs[0].buffer.ptr),
                    Routes {
                        ids: inputs[1].buffer.ptr,
                        weights: inputs[2].buffer.ptr,
                    },
                    producer.raw,
                    &mut || {},
                )?;
            }
            producer.drain()?;
            let mut bytes = vec![0; rows as usize * 5120 * 4];
            let output = CuteafdDeviceBuffer {
                bytes: bytes.len(),
                ptr: wave.output(),
                device_id: device.id,
                flags: 0,
            };
            lib.copy_d2h(&mut bytes, output)?;
            let values: Vec<_> = bytes
                .chunks_exact(4)
                .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
                .collect();
            ensure!(
                values.iter().all(|v| v.is_finite()) && values.iter().any(|v| *v != 0.),
                "invalid rank output"
            );
            eprintln!("single rank rows={rows} passed finite/nonzero launch");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires official CUTEAFD_SNAPSHOT, TP2 CUTEAFD_NATIVE_LIB and two GPUs"]
    fn real_encoder_rank_waves_share_weights_across_independent_lanes() -> Result<()> {
        let capacity: u32 = std::env::var("CUTEAFD_TP2_TEST_CAPACITY")
            .unwrap_or_else(|_| "16".into())
            .parse()?;
        ensure!(
            [1, 16, 80, 256, 1024, 4096].contains(&capacity),
            "invalid fixture capacity"
        );
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?),
        )?;
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
        let package = PathBuf::from(std::env::var("CUTEAFD_EXL3_AOT").unwrap_or_default());
        let weights = super::super::tp2_ffn::load_routed_weights(
            devices,
            &catalog,
            1,
            capacity,
            [4_000_000_000; 2],
            &package,
        )?;
        let mut first = OwnerPair::new(
            weights.clone(),
            capacity,
            cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
            false,
        )?;
        let mut second = OwnerPair::new(
            weights,
            capacity,
            cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
            false,
        )?;
        let mut inputs = Vec::new();
        for device in devices {
            inputs.push((
                Allocation::new(device, capacity as usize * 5280)?,
                Allocation::new(device, capacity as usize * 24)?,
                Allocation::new(device, capacity as usize * 24)?,
                Stream::new(device)?,
            ));
        }
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for (rows, nonzero) in [
            (capacity, false),
            (1, false),
            (capacity, true),
            (capacity.min(3), true),
            (1, true),
            (capacity, false),
        ] {
            let mut wire = vec![0u8; capacity as usize * 5280];
            for row in wire.chunks_exact_mut(5280) {
                row[..5120].fill(if nonzero { 0x38 } else { 0 });
                row[5120..].fill(127);
            }
            let ids: Vec<u8> = (0..capacity as i32 * 6)
                .flat_map(|i| (i % 6).to_ne_bytes())
                .collect();
            let routing: Vec<u8> = (0..capacity * 6)
                .flat_map(|_| (1f32 / 6.).to_ne_bytes())
                .collect();
            for (rank, (w, i, r, _)) in inputs.iter().enumerate() {
                devices[rank].run(|| {
                    lib.copy_h2d(w.buffer, &wire)?;
                    lib.copy_h2d(i.buffer, &ids)?;
                    lib.copy_h2d(r.buffer, &routing)
                })?;
            }
            let make_inputs = || {
                std::array::from_fn(|rank| OwnerInputs {
                    input: inputs[rank].0.buffer,
                    ids: inputs[rank].1.buffer,
                    weights: inputs[rank].2.buffer,
                    producer: &inputs[rank].3,
                })
            };
            let (x, y) = runtime.block_on(async {
                if std::env::var_os("CUTEAFD_TP2_TEST_SERIAL").is_some() {
                    let x = unsafe { first.execute(0, rows, 0, make_inputs()).await };
                    let y = unsafe { second.execute(0, rows, 1, make_inputs()).await };
                    (x, y)
                } else {
                    tokio::join!(
                        unsafe { first.execute(0, rows, 0, make_inputs()) },
                        unsafe { second.execute(0, rows, 1, make_inputs()) }
                    )
                }
            });
            eprintln!(
                "rows={rows} nonzero={nonzero} lane0={:?} lane1={:?}",
                x.as_ref().map(|_| ()),
                y.as_ref().map(|_| ())
            );
            let mut results = Vec::new();
            for (device, output) in [(devices[0], x?), (devices[1], y?)] {
                let mut host = vec![0; output.bytes];
                device.run(|| lib.copy_d2h(&mut host, output))?;
                assert!(host
                    .chunks_exact(2)
                    .all(|v| (u16::from_ne_bytes([v[0], v[1]]) & 0x7f80) != 0x7f80));
                assert_eq!(
                    host.chunks_exact(2)
                        .any(|v| u16::from_ne_bytes([v[0], v[1]]) & 0x7fff != 0),
                    nonzero
                );
                results.push(host);
            }
            let atomic = catalog.exl3().is_none()
                && catalog.nvfp4().is_none()
                && lib
                    .v41_tp2_expert_info(
                        crate::shared::experts::rtx::native::kernel_capacities(rows as usize)?
                            .last()
                            .copied()
                            .unwrap(),
                    )?
                    .abi_version
                    == 3;
            if atomic {
                // Independent atomic executions can straddle a BF16 rounding
                // boundary. Bound both each value and aggregate error; do not
                // dump multi-megabyte buffers when this fixture fails.
                let mut squared_error = 0f64;
                let mut squared_signal = 0f64;
                let mut changed = 0usize;
                for (a, b) in results[0].chunks_exact(2).zip(results[1].chunks_exact(2)) {
                    let a = u16::from_ne_bytes([a[0], a[1]]);
                    let b = u16::from_ne_bytes([b[0], b[1]]);
                    assert!(
                        a.abs_diff(b) <= 1,
                        "atomic output differs by more than one BF16 step: {a} vs {b}"
                    );
                    changed += usize::from(a != b);
                    let x = f32::from_bits((a as u32) << 16) as f64;
                    let y = f32::from_bits((b as u32) << 16) as f64;
                    squared_error += (x - y) * (x - y);
                    squared_signal += x * x;
                }
                let relative = (squared_error / squared_signal.max(f64::MIN_POSITIVE)).sqrt();
                assert!(relative < 1e-6, "atomic lane relative L2: {relative}");
                eprintln!("TP2 atomic rows={rows} changed={changed} relative_l2={relative}");
            } else {
                assert!(
                    results[0] == results[1],
                    "deterministic TP2 lane outputs differ at rows={rows}"
                );
            }
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        // Releasing one lane must not unload AOT modules retained by the other.
        drop(second);
        let make_inputs = || {
            std::array::from_fn(|rank| OwnerInputs {
                input: inputs[rank].0.buffer,
                ids: inputs[rank].1.buffer,
                weights: inputs[rank].2.buffer,
                producer: &inputs[rank].3,
            })
        };
        let output = runtime.block_on(unsafe { first.execute(0, capacity, 0, make_inputs()) })?;
        let mut host = vec![0; output.bytes];
        devices[0].run(|| lib.copy_d2h(&mut host, output))?;
        ensure!(
            host.chunks_exact(2)
                .all(|v| u16::from_ne_bytes([v[0], v[1]]) & 0x7fff == 0),
            "surviving lane lost its zero-input result"
        );
        assert_eq!(lib.cuda_get_device()?, 0);
        eprintln!("surviving lane after peer lane drop passed");
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA devices"]
    fn opposite_lane_peer_reductions_preserve_results_and_device() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let d0 = Device {
            library: &lib,
            id: 0,
        };
        let d1 = Device {
            library: &lib,
            id: 1,
        };
        let capacity = 16;
        let bytes = capacity * 5120 * 6 * 4;
        let a = Allocation::new(d0, bytes)?;
        let b = Allocation::new(d1, bytes)?;
        let p0 = Stream::new(d0)?;
        let p1 = Stream::new(d1)?;
        let mut lane0 = OwnerReduce::new(
            [d0, d1],
            Combine::OwnerReduce { owner: 0 },
            capacity as u32,
            cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
        )?;
        let mut lane1 = OwnerReduce::new(
            [d0, d1],
            Combine::OwnerReduce { owner: 1 },
            capacity as u32,
            cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
        )?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for (rows, token_sums, va, vb) in [
            (1, false, 1f32, 2f32),
            (16, true, 3., -1.),
            (16, false, -2., 4.),
        ] {
            let host_a: Vec<u8> = va.to_ne_bytes().into_iter().cycle().take(bytes).collect();
            let host_b: Vec<u8> = vb.to_ne_bytes().into_iter().cycle().take(bytes).collect();
            d0.run(|| lib.copy_h2d(a.buffer, &host_a))?;
            d1.run(|| lib.copy_h2d(b.buffer, &host_b))?;
            let (x, y) = runtime.block_on(async {
                tokio::join!(
                    unsafe {
                        lane0.reduce(
                            &a,
                            &b,
                            &p0,
                            &p1,
                            rows,
                            PartialDtype::F32,
                            if token_sums {
                                PartialLayout::Tokens
                            } else {
                                PartialLayout::Routes { topk: 6 }
                            },
                        )
                    },
                    unsafe {
                        lane1.reduce(
                            &b,
                            &a,
                            &p1,
                            &p0,
                            rows,
                            PartialDtype::F32,
                            if token_sums {
                                PartialLayout::Tokens
                            } else {
                                PartialLayout::Routes { topk: 6 }
                            },
                        )
                    }
                )
            });
            let expected = (((va + vb) * if token_sums { 1. } else { 6. }).to_bits() >> 16) as u16;
            for (device, output) in [(d0, x?), (d1, y?)] {
                let mut host = vec![0; output.bytes];
                device.run(|| lib.copy_d2h(&mut host, output))?;
                assert!(host
                    .chunks_exact(2)
                    .all(|v| u16::from_ne_bytes([v[0], v[1]]) == expected));
            }
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "tp2/reference_tests.rs"]
mod reference_tests;
