//! Opt-in real-checkpoint qualification of the shared native expert owner.
use super::*;
use crate::shared::{decode_graph::{fatal_drain, GraphOwner}, memory::LoadStream};
use cuteafd_core::FfnActivation;
use cuteafd_loader::RoutedExpertShape;

fn bf16(x: f32) -> f32 {
    let bits = u64::from(x.to_bits());
    f32::from_bits((((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16) as u32)
}

fn e4m3(code: u8) -> f32 {
    let sign = if code & 128 == 0 { 1.0 } else { -1.0 };
    let exponent = (code >> 3) & 15;
    let fraction = code & 7;
    if exponent == 0 { sign * f32::from(fraction) * 2.0f32.powi(-9) }
    else if exponent == 15 && fraction == 7 { f32::NAN }
    else { sign * (1.0 + f32::from(fraction) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7) }
}

const FP4: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
fn e2m1(code: u8) -> f32 { FP4[(code & 7) as usize] * if code & 8 == 0 { 1.0 } else { -1.0 } }

fn nearest_e4m3(value: f32) -> f32 {
    let mut best = (f32::INFINITY, 0u8);
    for code in 0..=126u8 {
        let distance = (e4m3(code) - value).abs();
        if distance < best.0 || (distance == best.0 && code & 1 == 0) { best = (distance, code); }
    }
    e4m3(best.1)
}

// Exhaustive finite-code nearest search is intentionally independent of GPU casts.
fn signed_e4m3_rne(value: f32) -> f32 {
    nearest_e4m3(value.abs().min(448.0)).copysign(value)
}

fn a8(input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut values = Vec::with_capacity(input.len());
    let mut scales = Vec::new();
    for block in input.chunks(128) {
        let scale = block.iter().copied().map(f32::abs).fold(1e-10f32, f32::max) * (1.0 / 448.0);
        let inverse = 1.0 / scale;
        scales.push(scale);
        values.extend(block.iter().map(|&x| signed_e4m3_rne((x * inverse).clamp(-448.0, 448.0))));
    }
    (values, scales)
}

fn a8_project(input: &[f32], weights: &[u8], scales: &[u8], n: usize, exact: bool) -> Vec<f32> {
    let k = input.len();
    let (q, a_scales) = a8(input);
    (0..n).map(|row| {
        let mut sum32 = 0.0f32;
        let mut sum64 = 0.0f64;
        for (kb, block) in q.chunks(128).enumerate() {
            let offset = (row / 128 * k.div_ceil(128) + kb) * 4;
            let ws = f32::from_le_bytes(scales[offset..offset + 4].try_into().unwrap());
            let mut partial32 = 0.0f32;
            let mut partial64 = 0.0f64;
            for (j, &a) in block.iter().enumerate() {
                let w = e4m3(weights[row * k + kb * 128 + j]);
                partial32 += a * w;
                partial64 += f64::from(a) * f64::from(w);
            }
            sum32 += (partial32 * a_scales[kb]) * ws;
            sum64 += partial64 * f64::from(a_scales[kb]) * f64::from(ws);
        }
        bf16(if exact { sum64 as f32 } else { sum32 })
    }).collect()
}

#[test]
fn native_a8_signed_rne_and_ragged_blocks() {
    for code in 0..126u8 {
        let a = e4m3(code);
        let b = e4m3(code + 1);
        let tie = (a + b) * 0.5;
        let expected = if code & 1 == 0 { a } else { b };
        assert_eq!(signed_e4m3_rne(tie), expected);
        assert_eq!(signed_e4m3_rne(-tie), -expected);
    }
    assert_eq!(signed_e4m3_rne(-0.0).to_bits(), (-0.0f32).to_bits());
    assert_eq!(signed_e4m3_rne(1000.0), 448.0);
    let mut x = vec![0.0; 129];
    x[128] = -3e-12;
    let (q, scales) = a8(&x);
    assert_eq!(q.len(), 129);
    assert_eq!(scales, vec![1e-10f32 * (1.0 / 448.0); 2]);
    assert!(q[128] < 0.0);
    x[0] = 3.25;
    let (_, scales) = a8(&x);
    assert_eq!(scales[0], 3.25f32 * (1.0 / 448.0));
    let weights = vec![56u8; 129];
    let scales = [0.0037f32.to_le_bytes(), 0.0013f32.to_le_bytes()].concat();
    assert_eq!(a8_project(&x, &weights, &scales, 1, true), a8_project(&x, &weights, &scales, 1, false));
}

// Quantize without folding input_scale into each element; the native GEMM
// applies that global factor once, after its FP32 accumulator.
fn a4(input: &[f32], input_scale: f32) -> Vec<f32> {
    let global = 1.0 / input_scale;
    input.chunks_exact(16).flat_map(|block| {
        let amax = block.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        let scale = nearest_e4m3((amax * global / 6.0).min(448.0));
        let value_scale = scale / global;
        block.iter().map(move |&value| {
            if scale == 0.0 { return 0.0; }
            let mut best = (f32::INFINITY, 0usize);
            for (code, level) in FP4.iter().enumerate() {
                let distance = (value.abs() - level * value_scale).abs();
                if distance < best.0 || (distance == best.0 && code & 1 == 0) { best = (distance, code); }
            }
            FP4[best.1] * scale * if value.is_sign_negative() { -1.0 } else { 1.0 }
        })
    }).collect()
}

fn project(tensors: &Fp8ExpertTensors, layer: usize, expert: usize, projection: Fp8Projection,
    tp: usize, rank: usize, slicing: Slicing, input: &[f32]) -> Result<Vec<f32>> {
    let width = tensors.rank_width(tp, rank, slicing)?;
    let hidden = tensors.shape().hidden;
    let (n, k) = if projection == Fp8Projection::Down { (hidden, width) } else { (width, hidden) };
    ensure!(input.len() == k, "CPU oracle input extent");
    let (weight_bytes, scale_bytes) = tensors.slice_bytes_with(projection, tp, rank, slicing)?;
    let mut weights = vec![0; weight_bytes];
    let mut scales = vec![0; scale_bytes];
    tensors.read_slice_with(layer, expert, projection, tp, rank, slicing,
        &mut weights, &mut scales, &mut Vec::new())?;
    let format = tensors.format();
    if tensors.arithmetic() == cuteafd_loader::formats::fp8_experts::ExpertArithmetic::K128F32A8 {
        return Ok(a8_project(input, &weights, &scales, n, true));
    }
    let input_scale = tensors.read_input_scale(layer, expert, projection)?;
    let alpha = tensors.read_alpha(layer, expert, projection)?;
    let quantized = if format == ExpertFormat::Nvfp4 { a4(input, input_scale) } else { input.to_vec() };
    let mut output = vec![0.0; n];
    for (row, output) in output.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        for (column, &activation) in quantized.iter().enumerate() {
            let at = row * k + column;
            let weight = match format {
                ExpertFormat::Bf16 => f32::from_bits(u32::from(u16::from_le_bytes(
                    weights[at * 2..at * 2 + 2].try_into().unwrap())) << 16),
                ExpertFormat::Fp8Block128 => {
                    let scale = (row / 128 * k.div_ceil(128) + column / 128) * 4;
                    bf16(e4m3(weights[at]) * f32::from_le_bytes(scales[scale..scale + 4].try_into().unwrap()))
                }
                ExpertFormat::Nvfp4 => e2m1((weights[at / 2] >> (4 * (at % 2))) & 15)
                    * e4m3(scales[row * (k / 16) + column / 16]),
                ExpertFormat::Mxfp4 => anyhow::bail!("MXFP4 is outside this qualification"),
            };
            sum += f64::from(activation) * f64::from(weight);
        }
        let accumulator = sum as f32;
        *output = bf16(if format == ExpertFormat::Nvfp4 { accumulator * (alpha * input_scale) } else { accumulator });
    }
    Ok(output)
}

fn oracle(tensors: &Fp8ExpertTensors, layer: usize, tp: usize, rank: usize,
    slicing: Slicing, input: &[f32]) -> Result<Vec<Vec<Vec<f32>>>> {
    let hidden = tensors.shape().hidden;
    let activation = tensors.activation(layer).context("checkpoint activation contract missing")?;
    ensure!(matches!(activation, FfnActivation::SiluPostClampFp32Product { .. }), "not post-SiLU");
    input.chunks_exact(hidden).map(|row| (0..8).map(|expert| {
        let gate = project(tensors, layer, expert, Fp8Projection::Gate, tp, rank, slicing, row)?;
        let up = project(tensors, layer, expert, Fp8Projection::Up, tp, rank, slicing, row)?;
        let act = gate.into_iter().zip(up).map(|(gate, up)| activation.evaluate(gate, up)).collect::<Vec<_>>();
        project(tensors, layer, expert, Fp8Projection::Down, tp, rank, slicing, &act)
    }).collect()).collect()
}

fn upload<'a>(library: &'a NativeLibrary, bytes: &[u8]) -> Result<DeviceAllocation<'a>> {
    let owner = DeviceAllocation::new(library, bytes.len())?;
    library.copy_h2d(owner.buffer, bytes)?;
    Ok(owner)
}

#[test]
fn cpu_format_and_quantizer_boundaries() {
    assert_eq!(e4m3(126), 448.0);
    assert_eq!(e4m3(1), 2.0f32.powi(-9));
    assert_eq!(e2m1(15), -6.0);
    assert_eq!(nearest_e4m3(1.0625), 1.0);
    assert_eq!(a4(&[0.0; 16], 0.01), vec![0.0; 16]);
    let values = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0,
        -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, -6.0];
    assert_eq!(a4(&values, 1.0), vec![0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0,
        -0.0, -1.0, -1.0, -2.0, -2.0, -4.0, -4.0, -6.0]);
}

#[test]
#[ignore = "requires CUDA, CUTEAFD_NATIVE_LIB and CUTEAFD_STEP5_{FORMAT,SNAPSHOT,PACKAGE,BF16_PACKAGE}"]
fn step5_checkpoint_mixed_native_execution_and_graph_replay() -> Result<()> {
    // SAFETY: this opt-in test accepts only operator-selected trusted build artifacts.
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    library.cuda_set_device(0)?;
    let rank = std::env::var("CUTEAFD_STEP5_RANK").unwrap_or_else(|_| "0".into()).parse::<usize>()?;
    let mut groups = Vec::new();
    let mut references = Vec::new();
    let format = match std::env::var("CUTEAFD_STEP5_FORMAT")?.as_str() {
        "fp8" => ExpertFormat::Fp8Block128,
        "nvfp4" => ExpertFormat::Nvfp4,
        other => anyhow::bail!("unsupported qualification format {other}"),
    };
    let snapshot = PathBuf::from(std::env::var("CUTEAFD_STEP5_SNAPSHOT")?);
    // Two distinct deterministic BF16 activation rows repeat at wider widths,
    // isolating row-width arithmetic and route-slot ordering from new inputs.
    let input = (0..8192).map(|i| bf16(((i * 17 % 257) as f32 - 128.0) / 32.0)).collect::<Vec<_>>();
    for (label, layer, expected_format, package) in [
        ("quantized", 3, format, "CUTEAFD_STEP5_PACKAGE"),
        ("BF16", 90, ExpertFormat::Bf16, "CUTEAFD_STEP5_BF16_PACKAGE"),
    ] {
        let directory = PathBuf::from(std::env::var(package)?);
        // SAFETY: trusted qualification package; metadata does not allocate GPU storage.
        let metadata = unsafe { Fp8MoeMetadata::read(&directory) }?;
        let tp = metadata.info.tp;
        let tensors = Fp8ExpertTensors::read(&snapshot, RoutedExpertShape { layers: 92, first_layer: 3,
            experts: 352, topk: 8, hidden: 4096, intermediate: 1536, draft_stages: 0, draft_experts: 0 })?;
        if label == "BF16" {
            let mixed = tensors.homogeneous_layers(&[3, 90])?;
            ensure!(mixed.len() == 2 && mixed[0].0.format() == format
                && mixed[1].0.format() == ExpertFormat::Bf16, "mixed checkpoint group dispatch");
        }
        let tensors = tensors.for_layer(layer)?;
        ensure!(tensors.format() == expected_format, "{label}: wrong checkpoint storage");
        let (directory, slicing) = exact_layout(&directory, &tensors, tp, rank);
        let metadata = package_metadata(&directory, &tensors, &[layer], tp, rank, slicing)?;
        ensure!(!metadata.info.wire_input, "qualification requires original BF16 activation input");
        references.push(oracle(&tensors, layer, tp, rank, slicing, &input)?);
        let budget = Fp8Layer::bytes_for(&tensors, tp, rank, slicing)? + metadata.scratch_for(64)?.max(256);
        groups.push(Fp8Experts::load(&library, &tensors, &directory, &[layer], tp, rank, 64, budget)?);
    }
    let all_input = (0..64).flat_map(|row| input[(row % 2) * 4096..(row % 2 + 1) * 4096].iter().copied())
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes()).collect::<Vec<_>>();
    let ids = (0..64 * 8).flat_map(|at| (((at / 8 + at % 8) % 8) as i32).to_le_bytes()).collect::<Vec<_>>();
    let routing = (0..64 * 8).flat_map(|at| ((at % 8 + 1) as f32 / 36.0).to_le_bytes()).collect::<Vec<_>>();
    let x = upload(&library, &all_input)?;
    let ids = upload(&library, &ids)?;
    let weights = upload(&library, &routing)?;
    let outputs = (0..groups.len()).map(|_| DeviceAllocation::new(&library, 64 * 4096 * 2)).collect::<Result<Vec<_>>>()?;
    let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
    for rows in [1, 16, 64] {
        let launch = || -> Result<()> {
            for (group, output) in groups.iter().zip(&outputs) {
                // SAFETY: buffers cover the full capacity and outlive this drained stream/graph.
                unsafe { group.run_bf16(0, rows, x.buffer.ptr, ids.buffer.ptr, weights.buffer.ptr,
                    output.buffer.ptr, stream.raw)?; }
            }
            Ok(())
        };
        let result = launch();
        // SAFETY: this stream is owned and all captured allocations remain alive.
        fatal_drain(unsafe { library.cuda_stream_synchronize(stream.raw) }.map_err(Into::into), "expert test warmup");
        result?;
        let mut eager = vec![vec![0; rows * 4096 * 2]; groups.len()];
        for (output, bytes) in outputs.iter().zip(&mut eager) { library.copy_d2h(bytes, output.buffer)?; }
        // SAFETY: only precompiled launches over stable allocations occur during capture.
        unsafe { library.cuda_graph_begin_capture(stream.raw)?; }
        let result = launch();
        // End capture even if a launch fails; retain/drain all graph resources on every exit.
        // SAFETY: begin_capture succeeded on this owned stream.
        let raw = unsafe { library.cuda_graph_end_capture(stream.raw)? };
        // SAFETY: external owners are in this scope and all replays are drained before drop.
        let graph = unsafe { GraphOwner::new(&library, 0, raw, ())? };
        result?;
        for _ in 0..3 {
            for output in &outputs { library.copy_h2d(output.buffer, &vec![0; 64 * 4096 * 2])?; }
            // SAFETY: the graph's owners and stream are alive, and previous replay is complete.
            let result = unsafe { library.cuda_graph_launch(graph.raw, stream.raw) };
            // SAFETY: owned stream; graph and its allocations remain retained through the drain.
            fatal_drain(unsafe { library.cuda_stream_synchronize(stream.raw) }.map_err(Into::into), "expert test replay");
            result?;
            for (group, output) in outputs.iter().enumerate() {
                let mut actual = vec![0; rows * 4096 * 2];
                library.copy_d2h(&mut actual, output.buffer)?;
                ensure!(actual == eager[group], "group {group} rows {rows}: graph changed bytes");
                let (mut dot, mut aa, mut bb, mut error) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for row in 0..rows {
                    let (mut rd, mut ra, mut rb) = (0.0f64, 0.0f64, 0.0f64);
                    for column in 0..4096 {
                        let mut expected = 0.0f32;
                        for slot in 0..8 {
                            expected += references[group][row % 2][(row + slot) % 8][column] * ((slot + 1) as f32 / 36.0);
                        }
                        let expected = f64::from(bf16(expected));
                        let at = (row * 4096 + column) * 2;
                        let value = f64::from(f32::from_bits(u32::from(u16::from_le_bytes(
                            actual[at..at + 2].try_into().unwrap())) << 16));
                        ensure!(value.is_finite() && expected.is_finite(), "nonfinite expert result");
                        rd += value * expected; ra += value * value; rb += expected * expected;
                        error += (value - expected).powi(2);
                    }
                    ensure!(ra > 0.0 && rb > 0.0 && rd / (ra * rb).sqrt() >= 0.9999,
                        "group {group} rows {rows} row {row}: CPU oracle cosine below 0.9999");
                    dot += rd; aa += ra; bb += rb;
                }
                let relative = (error / bb).sqrt();
                eprintln!("group {group} rows {rows}: cosine={} relative_l2={relative}", dot / (aa * bb).sqrt());
                ensure!(relative <= 0.02, "group {group} rows {rows}: CPU oracle relative L2 exceeds 2%");
            }
        }
    }
    Ok(())
}
