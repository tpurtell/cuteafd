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

// Dynamic K128/F32 A8 quantization: one F32 scale per 128-element K block
// (`max(amax, 1e-10) / 448.0`), reciprocal `1.0 / scale` in F32, RNE E4M3FN
// with finite saturation at +-448.0. Ragged tail blocks use only active lanes.
fn a8(input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut quantized = Vec::with_capacity(input.len());
    let mut scales = Vec::with_capacity(input.len().div_ceil(128));
    for block in input.chunks(128) {
        let amax = block.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        let scale = amax.max(1e-10_f32) * (1.0_f32 / 448.0_f32);
        let inv = 1.0_f32 / scale;
        scales.push(scale);
        for &value in block {
            let scaled = (value * inv).clamp(-448.0_f32, 448.0_f32);
            let q = nearest_e4m3(scaled.abs()) * if value.is_sign_negative() { -1.0 } else { 1.0 };
            quantized.push(q);
        }
    }
    (quantized, scales)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fp8OracleMode {
    QuantizedFp64,
    QuantizedFp32Mirror,
    UnquantizedFp64,
}

fn project_raw(format: ExpertFormat, n: usize, k: usize, weights: &[u8], scales: &[u8],
    input_scale: f32, alpha: f32, input: &[f32], mode: Fp8OracleMode) -> Result<Vec<f32>> {
    ensure!(input.len() == k, "CPU oracle input extent");
    let mut output = vec![0.0; n];
    if format == ExpertFormat::Fp8Block128 {
        let k_blocks = k.div_ceil(128);
        match mode {
            Fp8OracleMode::QuantizedFp32Mirror => {
                let (q_a, s_a) = a8(input);
                for (row, output) in output.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for (kb, &a_scale) in s_a.iter().enumerate() {
                        let mut part = 0.0f32;
                        for column in (kb * 128)..((kb + 1) * 128).min(k) {
                            part += q_a[column] * e4m3(weights[row * k + column]);
                        }
                        let scale = (row / 128 * k_blocks + kb) * 4;
                        let w_scale = f32::from_le_bytes(scales[scale..scale + 4].try_into().unwrap());
                        acc += part * (a_scale * w_scale);
                    }
                    *output = bf16(acc);
                }
            }
            Fp8OracleMode::QuantizedFp64 => {
                let (q_a, s_a) = a8(input);
                for (row, output) in output.iter_mut().enumerate() {
                    let mut sum = 0.0f64;
                    for (column, &q) in q_a.iter().enumerate() {
                        let at = row * k + column;
                        let scale = (row / 128 * k_blocks + column / 128) * 4;
                        let w_scale = f32::from_le_bytes(scales[scale..scale + 4].try_into().unwrap());
                        let activation = f64::from(q) * f64::from(s_a[column / 128]);
                        let weight = f64::from(e4m3(weights[at])) * f64::from(w_scale);
                        sum += activation * weight;
                    }
                    *output = bf16(sum as f32);
                }
            }
            Fp8OracleMode::UnquantizedFp64 => {
                for (row, output) in output.iter_mut().enumerate() {
                    let mut sum = 0.0f64;
                    for (column, &activation) in input.iter().enumerate() {
                        let at = row * k + column;
                        let scale = (row / 128 * k_blocks + column / 128) * 4;
                        let w_scale = f32::from_le_bytes(scales[scale..scale + 4].try_into().unwrap());
                        let weight = f64::from(e4m3(weights[at])) * f64::from(w_scale);
                        sum += f64::from(activation) * weight;
                    }
                    *output = bf16(sum as f32);
                }
            }
        }
        return Ok(output);
    }
    let quantized = if format == ExpertFormat::Nvfp4 { a4(input, input_scale) } else { input.to_vec() };
    for (row, output) in output.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        for (column, &activation) in quantized.iter().enumerate() {
            let at = row * k + column;
            let weight = match format {
                ExpertFormat::Bf16 => f32::from_bits(u32::from(u16::from_le_bytes(
                    weights[at * 2..at * 2 + 2].try_into().unwrap())) << 16),
                ExpertFormat::Nvfp4 => e2m1((weights[at / 2] >> (4 * (at % 2))) & 15)
                    * e4m3(scales[row * (k / 16) + column / 16]),
                ExpertFormat::Fp8Block128 | ExpertFormat::Mxfp4 => anyhow::bail!("MXFP4 is outside this qualification"),
            };
            sum += f64::from(activation) * f64::from(weight);
        }
        let accumulator = sum as f32;
        *output = bf16(if format == ExpertFormat::Nvfp4 { accumulator * (alpha * input_scale) } else { accumulator });
    }
    Ok(output)
}

fn project_with_mode(tensors: &Fp8ExpertTensors, layer: usize, expert: usize, projection: Fp8Projection,
    tp: usize, rank: usize, slicing: Slicing, input: &[f32], mode: Fp8OracleMode) -> Result<Vec<f32>> {
    let width = tensors.rank_width(tp, rank, slicing)?;
    let hidden = tensors.shape().hidden;
    let (n, k) = if projection == Fp8Projection::Down { (hidden, width) } else { (width, hidden) };
    let (weight_bytes, scale_bytes) = tensors.slice_bytes_with(projection, tp, rank, slicing)?;
    let mut weights = vec![0; weight_bytes];
    let mut scales = vec![0; scale_bytes];
    tensors.read_slice_with(layer, expert, projection, tp, rank, slicing,
        &mut weights, &mut scales, &mut Vec::new())?;
    let format = tensors.format();
    let input_scale = tensors.read_input_scale(layer, expert, projection)?;
    let alpha = tensors.read_alpha(layer, expert, projection)?;
    project_raw(format, n, k, &weights, &scales, input_scale, alpha, input, mode)
}

fn oracle_with_mode(tensors: &Fp8ExpertTensors, layer: usize, tp: usize, rank: usize,
    slicing: Slicing, input: &[f32], mode: Fp8OracleMode) -> Result<Vec<Vec<Vec<f32>>>> {
    let hidden = tensors.shape().hidden;
    let activation = tensors.activation(layer).context("checkpoint activation contract missing")?;
    ensure!(matches!(activation, FfnActivation::SiluPostClampFp32Product { .. }), "not post-SiLU");
    input.chunks_exact(hidden).map(|row| (0..8).map(|expert| {
        let gate = project_with_mode(tensors, layer, expert, Fp8Projection::Gate, tp, rank, slicing, row, mode)?;
        let up = project_with_mode(tensors, layer, expert, Fp8Projection::Up, tp, rank, slicing, row, mode)?;
        let act = gate.into_iter().zip(up).map(|(gate, up)| activation.evaluate(gate, up)).collect::<Vec<_>>();
        project_with_mode(tensors, layer, expert, Fp8Projection::Down, tp, rank, slicing, &act, mode)
    }).collect()).collect()
}

fn oracle(tensors: &Fp8ExpertTensors, layer: usize, tp: usize, rank: usize,
    slicing: Slicing, input: &[f32]) -> Result<Vec<Vec<Vec<f32>>>> {
    oracle_with_mode(tensors, layer, tp, rank, slicing, input, Fp8OracleMode::QuantizedFp64)
}

fn compare_reference(actual: &[u8], reference: &[Vec<Vec<f32>>], rows: usize) -> Result<(f64, f64)> {
    let (mut dot, mut aa, mut bb, mut error) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for row in 0..rows {
        let (mut rd, mut ra, mut rb) = (0.0f64, 0.0f64, 0.0f64);
        for column in 0..4096 {
            let mut expected = 0.0f32;
            for slot in 0..8 {
                expected += reference[row % 2][(row + slot) % 8][column] * ((slot + 1) as f32 / 36.0);
            }
            let expected = f64::from(bf16(expected));
            let at = (row * 4096 + column) * 2;
            let value = f64::from(f32::from_bits(u32::from(u16::from_le_bytes(
                actual[at..at + 2].try_into().unwrap())) << 16));
            ensure!(value.is_finite() && expected.is_finite(), "nonfinite expert result");
            rd += value * expected; ra += value * value; rb += expected * expected;
            error += (value - expected).powi(2);
        }
        ensure!(ra > 0.0 && rb > 0.0, "zero norm expert row {row}");
        dot += rd; aa += ra; bb += rb;
    }
    Ok((dot / (aa * bb).sqrt(), (error / bb).sqrt()))
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
    assert_eq!(nearest_e4m3(1.1875), 1.25);
    assert_eq!(nearest_e4m3(1000.0), 448.0);
    assert_eq!(a4(&[0.0; 16], 0.01), vec![0.0; 16]);
    let values = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0,
        -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, -6.0];
    assert_eq!(a4(&values, 1.0), vec![0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0,
        -0.0, -1.0, -1.0, -2.0, -2.0, -4.0, -4.0, -6.0]);

    // A8: zero block uses 1e-10 floor and produces all-zero E4M3 codes.
    let (q_zero, s_zero) = a8(&[0.0; 128]);
    assert_eq!(q_zero, vec![0.0; 128]);
    assert_eq!(s_zero, vec![1e-10_f32 * (1.0_f32 / 448.0_f32)]);

    // A8: tiny values below 1e-10 floor clamp scale to 1e-10 / 448.0; E4M3 subnormals round RNE.
    let mut tiny = vec![0.0f32; 128];
    tiny[0] = 1e-12;
    tiny[1] = -1e-12;
    let (q_tiny, s_tiny) = a8(&tiny);
    assert_eq!(s_tiny[0], 1e-10_f32 * (1.0_f32 / 448.0_f32));
    assert_eq!(q_tiny[0], nearest_e4m3(1e-12 / s_tiny[0]));
    assert_eq!(q_tiny[1], -nearest_e4m3(1e-12 / s_tiny[0]));

    // A8: ragged tail block, signs, saturation, and non-power-of-two scales.
    let mut ragged = vec![0.0f32; 130];
    ragged[0] = 3.1415927;
    ragged[1] = -3.1415927 * 0.5;
    ragged[2] = -0.0;
    ragged[128] = -17.5;
    ragged[129] = 8.75;
    let (q_ragged, s_ragged) = a8(&ragged);
    assert_eq!(s_ragged.len(), 2);
    assert_eq!(s_ragged[0], 3.1415927_f32 * (1.0_f32 / 448.0_f32));
    assert_eq!(s_ragged[1], 17.5_f32 * (1.0_f32 / 448.0_f32));
    assert_eq!(q_ragged[0], 448.0);
    assert_eq!(q_ragged[1], -224.0);
    assert!(q_ragged[2] == 0.0 && q_ragged[2].is_sign_negative());
    assert_eq!(q_ragged[128], -448.0);
    assert_eq!(q_ragged[129], 224.0);

    // Non-power-of-two weight and activation scales across two K128 blocks.
    let k = 256;
    let mut input = vec![0.0f32; k];
    for (i, v) in input.iter_mut().enumerate() {
        *v = bf16(((i as f32) - 100.0) * 0.073);
    }
    let weights: Vec<u8> = (0..k).map(|i| ((i * 13 + 7) % 120) as u8).collect();
    let scales: Vec<u8> = [0.03719_f32, 0.1137_f32].into_iter()
        .flat_map(f32::to_le_bytes).collect();
    let fp64_out = project_raw(ExpertFormat::Fp8Block128, 1, k, &weights, &scales,
        1.0, 1.0, &input, Fp8OracleMode::QuantizedFp64).unwrap();
    let fp32_out = project_raw(ExpertFormat::Fp8Block128, 1, k, &weights, &scales,
        1.0, 1.0, &input, Fp8OracleMode::QuantizedFp32Mirror).unwrap();
    assert_eq!(fp64_out, fp32_out);
}

#[test]
#[ignore = "requires CUDA, CUTEAFD_NATIVE_LIB and CUTEAFD_STEP5_{FORMAT,SNAPSHOT,PACKAGE,BF16_PACKAGE}"]
fn step5_checkpoint_mixed_native_execution_and_graph_replay() -> Result<()> {
    const QUAL_ROWS: [usize; 7] = [1, 7, 8, 9, 16, 64, 640];
    const MAX_ROWS: usize = 640;
    // SAFETY: this opt-in test accepts only operator-selected trusted build artifacts.
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    library.cuda_set_device(0)?;
    let rank = std::env::var("CUTEAFD_STEP5_RANK").unwrap_or_else(|_| "0".into()).parse::<usize>()?;
    let mut groups = Vec::new();
    let mut references = Vec::new();
    let mut f32_mirrors = Vec::new();
    let mut unquantized_baselines = Vec::new();
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
        if tensors.format() == ExpertFormat::Fp8Block128 {
            f32_mirrors.push(Some(oracle_with_mode(&tensors, layer, tp, rank, slicing, &input,
                Fp8OracleMode::QuantizedFp32Mirror)?));
            unquantized_baselines.push(Some(oracle_with_mode(&tensors, layer, tp, rank, slicing, &input,
                Fp8OracleMode::UnquantizedFp64)?));
        } else {
            f32_mirrors.push(None);
            unquantized_baselines.push(None);
        }
        let budget = Fp8Layer::bytes_for(&tensors, tp, rank, slicing)? + metadata.scratch_for(MAX_ROWS)?.max(256);
        groups.push(Fp8Experts::load(&library, &tensors, &directory, &[layer], tp, rank, MAX_ROWS, budget)?);
    }
    let all_input = (0..MAX_ROWS).flat_map(|row| input[(row % 2) * 4096..(row % 2 + 1) * 4096].iter().copied())
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes()).collect::<Vec<_>>();
    let ids = (0..MAX_ROWS * 8).flat_map(|at| (((at / 8 + at % 8) % 8) as i32).to_le_bytes()).collect::<Vec<_>>();
    let routing = (0..MAX_ROWS * 8).flat_map(|at| ((at % 8 + 1) as f32 / 36.0).to_le_bytes()).collect::<Vec<_>>();
    let x = upload(&library, &all_input)?;
    let ids = upload(&library, &ids)?;
    let weights = upload(&library, &routing)?;
    let outputs = (0..groups.len()).map(|_| DeviceAllocation::new(&library, MAX_ROWS * 4096 * 2)).collect::<Result<Vec<_>>>()?;
    let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
    let mut row0_baseline: Vec<Option<Vec<u8>>> = vec![None; groups.len()];
    for rows in QUAL_ROWS {
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
            for output in &outputs { library.copy_h2d(output.buffer, &vec![0; MAX_ROWS * 4096 * 2])?; }
            // SAFETY: the graph's owners and stream are alive, and previous replay is complete.
            let result = unsafe { library.cuda_graph_launch(graph.raw, stream.raw) };
            // SAFETY: owned stream; graph and its allocations remain retained through the drain.
            fatal_drain(unsafe { library.cuda_stream_synchronize(stream.raw) }.map_err(Into::into), "expert test replay");
            result?;
            for (group, output) in outputs.iter().enumerate() {
                let mut actual = vec![0; rows * 4096 * 2];
                library.copy_d2h(&mut actual, output.buffer)?;
                ensure!(actual == eager[group], "group {group} rows {rows}: graph changed bytes");
                let row0 = &actual[..4096 * 2];
                match &row0_baseline[group] {
                    Some(expected_row0) => ensure!(row0 == expected_row0.as_slice(),
                        "group {group} rows {rows}: row 0 output changed across batch widths"),
                    None => row0_baseline[group] = Some(row0.to_vec()),
                }
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
                let cosine = dot / (aa * bb).sqrt();
                let relative = (error / bb).sqrt();
                if let (Some(mirror), Some(unquant)) = (&f32_mirrors[group], &unquantized_baselines[group]) {
                    let (m_cos, m_rel) = compare_reference(&actual, mirror, rows)?;
                    let (u_cos, u_rel) = compare_reference(&actual, unquant, rows)?;
                    eprintln!("group {group} rows {rows}: fp64_oracle(cosine={cosine} relative_l2={relative}) f32_mirror(cosine={m_cos} relative_l2={m_rel}) unquantized_fp64(cosine={u_cos} relative_l2={u_rel})");
                } else {
                    eprintln!("group {group} rows {rows}: cosine={cosine} relative_l2={relative}");
                }
                ensure!(relative <= 0.02, "group {group} rows {rows}: CPU oracle relative L2 exceeds 2%");
            }
        }
    }
    Ok(())
}
