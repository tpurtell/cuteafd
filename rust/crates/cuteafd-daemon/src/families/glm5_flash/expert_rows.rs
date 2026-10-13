//! Fixed-input expert-row checks through the production local packages.
use super::{bf16s, similarity, EngineArgs, Opened};
use crate::families::deepseek_v4::local::{LocalExperts, LocalLayer};
use crate::shared::memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};

#[derive(serde::Deserialize)]
struct Snapshot {
    layer: usize,
    rows: usize,
    hidden: usize,
    topk: usize,
    wire: std::path::PathBuf,
    ids: std::path::PathBuf,
    weights: std::path::PathBuf,
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    let ordered = |x: f32| { let bits = x.to_bits();
        if bits & 0x8000_0000 != 0 { !bits } else { bits | 0x8000_0000 } };
    if a == b { 0 } else { ordered(a).abs_diff(ordered(b)) }
}

fn snapshot_rows(path: &std::path::Path, name: &std::path::Path, rows: usize, width: usize) -> Result<Vec<u8>> {
    ensure!(name.components().count() == 1 && matches!(name.components().next(),
        Some(std::path::Component::Normal(_))), "snapshot buffer must be a sibling filename");
    let mut data = std::fs::read(path.parent().context("snapshot parent")?.join(name))?;
    let bytes = rows.checked_mul(width).context("snapshot size overflow")?;
    ensure!(data.len() >= bytes, "snapshot buffer is shorter than its live rows");
    data.truncate(bytes);
    Ok(data)
}

pub(super) fn check_snapshot(args: &EngineArgs, opened: &Opened, path: &std::path::Path) -> Result<()> {
    use crate::shared::{experts::rtx::{exl3::Exl3Tp2, ExpertInput, Routes, RtxExpertLayer}, memory::device::Device};
    let snapshot: Snapshot = serde_json::from_slice(&std::fs::read(path)?)?;
    let catalog = opened.experts.as_ref().context("expert snapshot requires --local-experts")?;
    let manifest = catalog.exl3().context("expert snapshot requires an EXL3 checkpoint")?;
    let shape = catalog.routed_experts();
    ensure!((2..=16).contains(&snapshot.rows) && snapshot.hidden == shape.hidden && snapshot.topk == shape.topk
        && snapshot.layer >= shape.first_layer && snapshot.layer < opened.cfg.layers,
        "snapshot geometry does not match checkpoint");
    let (rows, hidden, topk) = (snapshot.rows, snapshot.hidden, snapshot.topk);
    let stride = hidden + hidden / 32;
    let wire = snapshot_rows(path, &snapshot.wire, rows, stride)?;
    let ids = snapshot_rows(path, &snapshot.ids, rows, topk * 4)?;
    let weights = snapshot_rows(path, &snapshot.weights, rows, topk * 4)?;
    ensure!(ids.chunks_exact(4).all(|b| u32::from_le_bytes(b.try_into().unwrap()) < shape.experts as u32),
        "snapshot contains out-of-range expert ids");
    ensure!(weights.chunks_exact(4).all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
        "snapshot contains non-finite route weights");
    let library = &opened.library;
    let devices = [Device { library, id: args.device }, Device { library,
        id: args.split_device.context("expert snapshot requires --split-device")? }];
    ensure!(devices[0].id != devices[1].id, "expert snapshot requires two distinct GPUs");
    let budgets = devices.map(|d| d.run(|| Ok(library.cuda_memory_info()?.0.saturating_sub(4 << 30))));
    let budgets = [*budgets[0].as_ref().map_err(|e| anyhow::anyhow!("{e}"))?,
        *budgets[1].as_ref().map_err(|e| anyhow::anyhow!("{e}"))?];
    let package = crate::shared::experts::exl3::aot_layout_directory(&args.native_lib, manifest.decoder_tiers(), "rtx-tp2");
    let mut experts = Exl3Tp2::load_pair(devices, catalog, &package,
        snapshot.layer..snapshot.layer + 1, rows, budgets)?;
    for (rank, device) in devices.into_iter().enumerate() {
        device.run(|| -> Result<()> {
            let stream = LoadStream { library, raw: library.cuda_stream_create()? };
            let inputs = [DeviceAllocation::new(library, wire.len())?, DeviceAllocation::new(library, ids.len())?,
                DeviceAllocation::new(library, weights.len())?];
            for (dst, src) in inputs.iter().zip([&wire, &ids, &weights]) { library.copy_h2d(dst.buffer, src)?; }
            let mut launch = |first: usize, count: usize| -> Result<Vec<f32>> {
                // SAFETY: all subrows are in the rank-owned inputs, retained until the isolated
                // stream drains; this diagnostic has no peer waiters requiring abort publication.
                unsafe { experts[rank].enqueue(snapshot.layer, count,
                    ExpertInput::Fp8K32(inputs[0].buffer.ptr.wrapping_byte_add(first * stride)),
                    Routes { ids: inputs[1].buffer.ptr.wrapping_byte_add(first * topk * 4),
                        weights: inputs[2].buffer.ptr.wrapping_byte_add(first * topk * 4) },
                    stream.raw, &mut || {})?; library.cuda_stream_synchronize(stream.raw)?; }
                let mut bytes = vec![0; count * hidden * 4];
                library.copy_d2h(&mut bytes, cuteafd_ffi::CuteafdDeviceBuffer { ptr: experts[rank].output(),
                    bytes: count * hidden * 4, device_id: device.id, flags: 0 })?;
                let values: Vec<_> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                ensure!(values.iter().all(|v| v.is_finite()), "expert output is non-finite");
                Ok(values)
            };
            let result = (|| -> Result<()> {
                let wide = launch(0, rows)?;
                let repeated = launch(0, rows)?;
                ensure!(wide.iter().zip(&repeated).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "repeated wide expert output changed");
                let captured = path.parent().context("snapshot parent")?.join(format!("routed_rank{rank}.bin"));
                if captured.is_file() {
                    let captured = std::fs::read(captured)?;
                    let actual: Vec<_> = wide.iter().flat_map(|v| v.to_le_bytes()).collect();
                    println!("layer {}, rank {rank}, m80 matches captured production partial: {}",
                        snapshot.layer, actual == captured);
                }
                for row in 0..rows {
                    let serial = launch(row, 1)?;
                    let together = &wide[row * hidden..(row + 1) * hidden];
                    let mut ulps: Vec<_> = serial.iter().zip(together).map(|(&a, &b)| ulp_distance(a, b)).collect();
                    ulps.sort_unstable();
                    let max = serial.iter().zip(together).map(|(&a, &b)| (a as f64 - b as f64).abs()).fold(0f64, f64::max);
                    let different = ulps.iter().filter(|&&x| x != 0).count();
                    println!("layer {}, rank {rank}, row {row}, m1 vs m80: different {different}/{hidden}, max_abs {max:.9e}, FP32 ulps p50 {} p90 {} p99 {} max {}",
                        snapshot.layer, ulps[hidden/2], ulps[hidden*90/100], ulps[hidden*99/100], ulps[hidden-1]);
                }
                Ok(())
            })();
            // SAFETY: drain even on error before releasing any launch input or output owner.
            unsafe { library.cuda_stream_synchronize(stream.raw)?; }
            result
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    #[test]
    fn ulps_are_sign_aware_and_identical_zeros_match() {
        assert_eq!(ulp_distance(0.0, -0.0), 0);
        assert_eq!(ulp_distance(1.0, f32::from_bits(1f32.to_bits() + 1)), 1);
        assert_eq!(ulp_distance(-1.0, f32::from_bits((-1f32).to_bits() + 1)), 1);
        assert_eq!(ulp_distance(-1.0, -1.0), 0);
    }
    #[test]
    fn snapshot_paths_cannot_escape_the_private_directory() {
        for name in ["../wire.bin", "/wire.bin", "sub/wire.bin", "."] {
            assert!(snapshot_rows(std::path::Path::new("/unused/expert.json"), std::path::Path::new(name), 6, 4224).is_err());
        }
    }
}

pub(super) fn check(args: &EngineArgs, opened: &Opened, max_rows: usize) -> Result<()> {
    ensure!(
        (2..=super::engine::DECODE_ROWS).contains(&max_rows),
        "--expert-row-check needs 2..=64 rows"
    );
    let catalog = opened
        .experts
        .as_ref()
        .context("--expert-row-check needs --local-experts")?;
    ensure!(
        catalog.exl3().is_some(),
        "--expert-row-check needs an EXL3/tr3 checkpoint"
    );
    let (library, shape) = (&opened.library, catalog.routed_experts());
    let stream = LoadStream {
        library,
        raw: library.cuda_stream_create()?,
    };
    let (free, _) = library.cuda_memory_info()?;
    let layer = shape.first_layer;
    let mut experts = LocalExperts::load_range(
        library,
        &args.native_lib,
        catalog,
        0,
        layer..layer + 1,
        max_rows,
        free.saturating_sub(4 << 30),
        stream.raw,
    )?
    .context("no local expert package for this checkpoint")?;
    let (hidden, topk, count) = (shape.hidden, shape.topk, shape.experts);
    let stride = hidden + hidden / 32;
    let wire = DeviceAllocation::new(library, max_rows * stride)?;
    let ids = DeviceAllocation::new(library, max_rows * topk * 4)?;
    let weights = DeviceAllocation::new(library, max_rows * topk * 4)?;
    let shared = DeviceAllocation::new(library, max_rows * hidden * 2)?;
    library.copy_h2d(shared.buffer, &vec![0; shared.buffer.bytes])?;
    library.copy_h2d(
        weights.buffer,
        &(0..max_rows * topk)
            .flat_map(|_| (1f32 / topk as f32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    // Representable E4M3 values around unit scale with UE8M0 scale 1 per K32.
    let input: Vec<u8> = (0..max_rows)
        .flat_map(|r| {
            (0..stride).map(move |c| {
                if c >= hidden {
                    127
                } else {
                    let bits = (r as u32)
                        .wrapping_mul(1664525)
                        .wrapping_add((c as u32).wrapping_mul(1013904223));
                    0x20 + (bits >> 24) as u8 % 32 | (bits as u8 & 0x80)
                }
            })
        })
        .collect();
    let mut widths = vec![
        1,
        4.min(max_rows),
        9.min(max_rows),
        16.min(max_rows),
        17.min(max_rows),
        max_rows,
    ];
    widths.sort_unstable();
    widths.dedup();
    let result = (|| -> Result<()> {
        for (high_ids, shared_routes) in
            [(false, true), (false, false), (true, true), (true, false)]
        {
            let pattern = format!(
                "{}{}",
                if shared_routes { "shared" } else { "diverse" },
                if high_ids { " high-ID" } else { "" }
            );
            let routes: Vec<u8> = (0..max_rows)
                .flat_map(|r| {
                    (0..topk).flat_map(move |j| {
                        let expert = if high_ids { count - topk + j } else { j * 31 };
                        (((expert + if shared_routes { 0 } else { r * 13 }) % count) as u32)
                            .to_le_bytes()
                    })
                })
                .collect();
            library.copy_h2d(ids.buffer, &routes)?;
            let mut launch = |rows: usize, input: &[u8]| -> Result<Vec<u8>> {
                library.copy_h2d(wire.buffer, input)?;
                // SAFETY: live wire/routes/shared allocations cover `rows`;
                // only this stream uses the bound production expert workspace.
                unsafe {
                    experts.run(
                        LocalLayer::Backbone(layer),
                        rows,
                        wire.buffer.ptr,
                        ids.buffer.ptr,
                        weights.buffer.ptr,
                        shared.buffer.ptr,
                        stream.raw,
                    )?
                };
                // SAFETY: this stream owns the launch and all its buffers stay live.
                unsafe { library.cuda_stream_synchronize(stream.raw)? };
                let mut output = vec![0u8; hidden * 2];
                library.copy_d2h(
                    &mut output,
                    cuteafd_ffi::CuteafdDeviceBuffer {
                        bytes: hidden * 2,
                        ..experts.output.buffer
                    },
                )?;
                ensure!(
                    bf16s(&output).iter().all(|x| x.is_finite()) && output.iter().any(|&b| b != 0),
                    "{pattern} routes, {rows} rows: non-finite or empty expert output"
                );
                Ok(output)
            };
            let serial = launch(1, &input)?;
            for &rows in &widths {
                let baseline = launch(rows, &input)?;
                ensure!(
                    baseline == launch(rows, &input)?,
                    "{pattern} routes, {rows} rows: identical launches differ"
                );
                let mut changed = input.clone();
                for row in changed[stride..].chunks_exact_mut(stride) {
                    row[..hidden].iter_mut().for_each(|x| *x ^= 0x80);
                }
                ensure!(
                    baseline == launch(rows, &changed)?,
                    "{pattern} routes, {rows} rows: later inputs changed the first expert row"
                );
                let a = bf16s(&serial);
                let b = bf16s(&baseline);
                let (cosine, relative) = similarity(&a, &b);
                let worst = a
                    .iter()
                    .zip(&b)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let different = serial
                    .chunks_exact(2)
                    .zip(baseline.chunks_exact(2))
                    .filter(|(a, b)| a != b)
                    .count();
                println!("layer {layer}, {pattern} routes, {rows} rows: first row vs serial {different}/{hidden} \
                    BF16 values differ, cosine {cosine:.9}, relative L2 {relative:.6e}, max |delta| {worst:.6e}; \
                    later-input causality exact");
            }
        }
        Ok(())
    })();
    // SAFETY: even an error after a launch must drain before its input and
    // expert storage drops. The stream owner also drains on destruction.
    unsafe { library.cuda_stream_synchronize(stream.raw)? };
    result
}
