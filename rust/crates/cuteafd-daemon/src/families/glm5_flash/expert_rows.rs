//! Fixed-input expert-row checks through the production local packages.
use super::{bf16s, similarity, EngineArgs, Opened};
use crate::families::deepseek_v4::local::{LocalExperts, LocalLayer};
use crate::shared::memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};

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
    let mut experts = LocalExperts::load(
        library,
        &args.native_lib,
        catalog,
        0,
        &[layer],
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
