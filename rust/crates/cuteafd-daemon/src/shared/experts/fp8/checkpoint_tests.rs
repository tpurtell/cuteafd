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
                ExpertFormat::Mxfp4 | ExpertFormat::Mxfp8 => anyhow::bail!("MX formats are outside this qualification"),
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
    let owner = DeviceAllocation::new(library, bytes.len(), crate::shared::memory::TEST)?;
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
#[ignore = "requires CUDA, CUTEAFD_NATIVE_LIB and a BF16-input FP8 CUTEAFD_FP8_SLOT_PACKAGE"]
fn native_fp8_two_stream_slots_are_byte_exact_against_serial() -> Result<()> {
    // SAFETY: this opt-in test accepts only operator-selected trusted build artifacts.
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    library.cuda_set_device(0)?;
    let directory = PathBuf::from(std::env::var("CUTEAFD_FP8_SLOT_PACKAGE")?);
    // SAFETY: the selected native package targets the current CUDA device.
    let module = unsafe { Fp8MoeModule::load(&directory)? };
    let info = module.info().clone();
    ensure!(info.weights == Fp8MoeWeights::Fp8 && !info.wire_input,
        "two-stream qualification requires native block-FP8 weights with BF16 input");
    let capacity = 256usize;
    ensure!(info.capacity_for(capacity).is_some(), "test package needs 256 rows");
    let mut regions = Vec::new();
    for (projection, n, k) in [(0, info.slice, info.hidden), (1, info.slice, info.hidden),
        (2, info.hidden, info.slice)] {
        let matrix = n.checked_mul(k).context("test matrix overflow")?;
        let weights = (0..info.experts * matrix).map(|at| {
            let expert = at / matrix;
            0x20 + ((at + expert * 7 + projection * 3) % 8) as u8
        }).collect::<Vec<_>>();
        let scales = (0..info.experts * n.div_ceil(128) * k.div_ceil(128))
            .flat_map(|at| (0.01f32 + (at % 3) as f32 * 0.001).to_le_bytes()).collect::<Vec<_>>();
        regions.push(Some(upload(&library, &weights)?.into()));
        regions.push(Some(upload(&library, &scales)?.into()));
    }
    let resident = Rc::new(Fp8Weights { layers: vec![Fp8Layer { layer: 0,
        slice: cuteafd_core::SparkSlice { unit_map: None, layer: 0, first: 0,
            width: info.slice as u32, stored_width: info.slice as u32,
            tiers: vec![], projection_counts: vec![] }, regions }],
        module, bf16_module: None, tp: info.tp, rank: 0, weight_read_bytes: 1 });
    let slots = [Fp8Experts::from_shared(&library, resident.clone(), capacity)?,
        Fp8Experts::from_shared(&library, resident.clone(), capacity)?];
    ensure!(slots[0].scratch.buffer.ptr != slots[1].scratch.buffer.ptr, "slots alias scratch");
    let inputs = (0..2).map(|slot| {
        let bytes = (0..capacity * info.hidden).flat_map(|at| {
            let value = bf16(((at * 17 % 127) as f32 + 1.0) / (128.0 + slot as f32 * 128.0));
            ((value.to_bits() >> 16) as u16).to_le_bytes()
        }).collect::<Vec<_>>();
        upload(&library, &bytes)
    }).collect::<Result<Vec<_>>>()?;
    let ids = upload(&library, &(0..capacity * info.topk)
        .flat_map(|at| ((at % info.experts) as i32).to_le_bytes()).collect::<Vec<_>>())?;
    let routing = upload(&library, &(0..capacity * info.topk)
        .flat_map(|_| (1.0f32 / info.topk as f32).to_le_bytes()).collect::<Vec<_>>())?;
    let outputs = (0..2).map(|_| DeviceAllocation::new(&library, capacity * info.hidden * 2, crate::shared::memory::TEST))
        .collect::<Result<Vec<_>>>()?;
    // Declared last: these streams drain before any slot, weights or I/O is freed on every exit.
    let streams = (0..2).map(|_| Ok(LoadStream { library: &library, raw: library.cuda_stream_create()? }))
        .collect::<Result<Vec<_>>>()?;
    for rows in [16usize, capacity] {
        let launch = |slot: usize| -> Result<()> {
            // SAFETY: this execution owns scratch; all buffers cover capacity and outlive both streams.
            unsafe { slots[slot].run_bf16(0, rows, inputs[slot].buffer.ptr, ids.buffer.ptr,
                routing.buffer.ptr, outputs[slot].buffer.ptr, streams[slot].raw) }
        };
        let mut serial = vec![vec![0; rows * info.hidden * 2]; 2];
        for slot in 0..2 {
            launch(slot)?;
            // SAFETY: owned stream; all its allocations remain live.
            unsafe { library.cuda_stream_synchronize(streams[slot].raw)?; }
            streams[slot].copy_d2h(&mut serial[slot], outputs[slot].buffer)?;
            ensure!(serial[slot].iter().any(|&byte| byte != 0), "zero serial output");
        }
        ensure!(serial[0] != serial[1], "test inputs do not distinguish the executions");
        for iteration in 0..32 {
            launch(iteration % 2)?;
            launch(1 - iteration % 2)?;
            for stream in &streams {
                // SAFETY: all queued work still owns its scratch, weights and I/O.
                unsafe { library.cuda_stream_synchronize(stream.raw)?; }
            }
            for slot in 0..2 {
                let mut actual = vec![0; serial[slot].len()];
                streams[slot].copy_d2h(&mut actual, outputs[slot].buffer)?;
                ensure!(actual == serial[slot], "rows {rows} iteration {iteration} slot {slot}: concurrent bytes differ");
            }
        }
        eprintln!("rows {rows}: two private-scratch streams byte-identical to serial over 32 overlaps");
    }
    Ok(())
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
            experts: 352, topk: 8, hidden: 4096, intermediate: 1536, draft_stages: 0, draft_experts: 0, activation: cuteafd_core::ExpertActivation::SiluPostClamp7 })?;
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
    let outputs = (0..groups.len()).map(|_| DeviceAllocation::new(&library, 64 * 4096 * 2, crate::shared::memory::TEST)).collect::<Result<Vec<_>>>()?;
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
        for (output, bytes) in outputs.iter().zip(&mut eager) { stream.copy_d2h(bytes, output.buffer)?; }
        // SAFETY: only precompiled launches over stable allocations occur during capture.
        unsafe { library.cuda_graph_begin_capture(stream.raw)?; }
        let result = launch();
        // End capture even if a launch fails; retain/drain all graph resources on every exit.
        // SAFETY: begin_capture succeeded on this owned stream.
        let raw = unsafe { library.cuda_graph_end_capture_for_test(stream.raw)? };
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
                stream.copy_d2h(&mut actual, output.buffer)?;
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

/// Deterministic NVFP4 operand for `experts` x `[n, k]`: packed E2M1 codes,
/// positive E4M3 block scales, then FP32 alpha and input-scale per expert,
/// laid out as the installed package reads it (fork `_nvfp4_moe_a4`).
fn nvfp4_operand(experts: usize, n: usize, k: usize, salt: usize, divisor: bool) -> (Vec<u8>, Vec<u8>) {
    let packed = (0..experts * n * k / 2).map(|at| ((at * 37 + salt * 11) % 251) as u8).collect::<Vec<_>>();
    let mut scales = (0..experts * n * k / 16).map(|at| (4 + (at * 7 + salt) % 7) as u8 * 8 + ((at + salt) % 8) as u8)
        .collect::<Vec<_>>();
    for expert in 0..experts {
        let alpha = 1e-3f32 + (expert % 5) as f32 * 4e-4;
        scales.extend((if divisor { 1.0 / alpha } else { alpha }).to_le_bytes());
    }
    for expert in 0..experts {
        let input = (5.0f32 / (6.0 * 448.0)) * 2f32.powi((expert % 7) as i32 - 3);
        scales.extend((if divisor { 1.0 / input } else { input }).to_le_bytes());
    }
    (packed, scales)
}

/// Merged-wave correctness of the shared Spark staging over a real installed
/// row-addressed NVFP4 A4 package: units staged from mapped frames and flushed
/// as one wave must write every member's reply byte-identical to that unit run
/// alone through the same worker (#2191). One Spark; the package is any
/// `fp8-<family>-nvfp4a4/<layout>` Spark export with a segmented program,
/// over BF16 rows or exact NVFP4 wire rows (`nvfp4_input`). Expert timing is
/// on: one span per singleton and per merged wave.
#[test]
#[ignore = "requires an SM121 Spark, CUTEAFD_NATIVE_LIB and a Spark NVFP4 A4 CUTEAFD_NVFP4_SEGMENT_PACKAGE"]
fn nvfp4_merged_waves_are_byte_identical_to_singleton_units() -> Result<()> {
    use crate::shared::experts::execution::HostExpertExchange;
    use crate::shared::memory::HostAllocation;
    use cuteafd_transport::{expert::{BackboneRequest, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16},
        ExpertProtocolV2Request, ExpertProtocolV2RequestView, ExpertProtocolV2RowDescriptor,
        ExpertProtocolV2RouteEntry, ExpertV2Dtype, ExpertV2SourceKind, ProtocolV2RequestDevicePayload,
        EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN};
    let directory = PathBuf::from(std::env::var("CUTEAFD_NVFP4_SEGMENT_PACKAGE")?);
    // The process geometry is fixed before anything reads it (library load and
    // package metadata do): the family whose routed shape the package's own
    // manifest (one directory up) declares for this layout.
    let layout = directory.file_name().and_then(|n| n.to_str()).context("package layout directory")?.to_owned();
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("../manifest.json"))?)?;
    let declared = &manifest["layouts"][layout.as_str()];
    let field = |key: &str| declared[key].as_u64().map(|v| v as u32).with_context(|| format!("manifest {key}"));
    use cuteafd_core::ExpertGeometry as G;
    let (hidden, experts, topk, intermediate) = (field("hidden")?, field("experts")?, field("top_k")?, field("intermediate")?);
    let shape = [G::MINIMAX_M3, G::GLM5_FLASH, G::GLM5, G::QWEN4, G::STEP5, G::INKLING2].into_iter()
        .find(|g| (g.hidden, g.experts, g.topk, g.intermediate) == (hidden, experts, topk, intermediate))
        .context("package shape is no known family")?;
    cuteafd_core::set_expert_geometry(shape)
        .map_err(|set| anyhow::anyhow!("process geometry already {set:?}; run this test alone"))?;
    // SAFETY: this opt-in test accepts only operator-selected trusted build artifacts.
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    library.cuda_set_device(0)?;
    // SAFETY: the selected native package targets the current CUDA device.
    let module = unsafe { Fp8MoeModule::load(&directory)? };
    let info = module.info().clone();
    ensure!(info.weights == Fp8MoeWeights::Nvfp4 && !info.wire_input && module.segment_recipe().is_some(),
        "merged-wave oracle requires a BF16- or NVFP4-input NVFP4 package with a row-addressed program");
    let nvfp4_wire = info.nvfp4_input;
    let (hidden, topk, experts) = (info.hidden, info.topk, info.experts);
    let divisor = info.scale_convention == cuteafd_ffi::fp8_moe::Fp8MoeScaleConvention::Divisor;
    let mut regions = Vec::new();
    for (salt, (n, k)) in [(info.slice, hidden), (info.slice, hidden), (hidden, info.slice)].into_iter().enumerate() {
        let (packed, scales) = nvfp4_operand(experts, n, k, salt, divisor);
        regions.push(Some(upload(&library, &packed)?.into()));
        regions.push(Some(upload(&library, &scales)?.into()));
    }
    let weights = Rc::new(Fp8Weights { layers: vec![Fp8Layer { layer: 3,
        slice: cuteafd_core::SparkSlice { unit_map: None, layer: 3, first: 0, width: info.slice as u32,
            stored_width: info.slice as u32, tiers: vec![], projection_counts: vec![] }, regions }],
        module, bf16_module: None, tp: info.tp, rank: 0, weight_read_bytes: 1 });
    // The serving unit (2752 rows), three buffered; one merged launch holds two units.
    let unit = 2752usize;
    let capacity = info.capacity_for(unit).context("package lacks a covering capacity")?;
    let extent = cuteafd_core::OwnerExtent { owner: cuteafd_core::ExecutionOwner::Prefill,
        capacity: unit as u32, compiled: capacity as u32, staging_rows: unit as u32, buffered_units: 3 };
    let timing_dir = std::env::temp_dir().join(format!("nvfp4-wave-timing-{}", std::process::id()));
    std::fs::create_dir_all(&timing_dir)?;
    std::env::set_var("CUTEAFD_EXPERT_TIMING", "1");
    std::env::set_var("CUTEAFD_PROFILE_DIR", &timing_dir);
    std::env::set_var("CUTEAFD_PROFILE_HOST", "oracle");
    let scratch = Rc::new(DeviceAllocation::new(&library, weights.owner_scratch_bytes(extent)?, crate::shared::memory::TEST)?);
    let mut worker = worker::Fp8Worker::with_extent(&library, weights.clone(), unit as u32, Some(scratch), Some(extent))?;
    worker.prepare_waves(extent)?;
    worker.defer(3, unit, cuteafd_core::ExecutionOwner::Prefill)?;
    worker.bind_layer(0)?;
    let executor = worker.executor_id();
    let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
    // Members: full units, uneven units and a short tail; hot experts make
    // multi-chunk groups whose rows interleave across members.
    let rows = [2752u32, 2752, 2487, 1900, 333];
    let mut frames = Vec::new();
    let mut replies = Vec::new();
    for (unit_index, &count) in rows.iter().enumerate() {
        // Slots 0 and 1 hit two hot experts (multi-chunk groups shared by
        // every member); the rest walk distinct experts per row.
        let ids = (0..count as usize * topk).map(|route| {
            let (row, slot) = (route / topk, route % topk);
            if slot < 2 { slot as u32 } else {
                (2 + (row * 13 + unit_index * 29 + (slot - 2) * ((experts - 2) / topk).max(1)) % (experts - 2)) as u32 }
        }).collect::<Vec<_>>();
        for row in ids.chunks_exact(topk) {
            let mut sorted = row.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            ensure!(sorted.len() == topk, "oracle routes repeat an expert within a row");
        }
        // Exact NVFP4 wire rows: packed E2M1, finite E4M3 K16 block scales,
        // a positive F32 row amax and 12 zero bytes; otherwise BF16 rows.
        let row_bytes = cuteafd_core::nvfp4_wire_row_bytes(hidden);
        let payload: Vec<u8> = if nvfp4_wire {
            (0..count as usize).flat_map(|row| {
                let mut bytes = vec![0u8; row_bytes];
                for (at, value) in bytes[..hidden / 2].iter_mut().enumerate() {
                    *value = ((at * 37 + row * 11 + unit_index * 53) % 251) as u8;
                }
                for (group, scale) in bytes[hidden / 2..hidden / 2 + hidden / 16].iter_mut().enumerate() {
                    *scale = 0x30 + ((row + group + unit_index) % 16) as u8;
                }
                let amax = 1.0f32 + ((row * 7 + unit_index) % 13) as f32 / 8.0;
                bytes[hidden / 2 + hidden / 16..hidden / 2 + hidden / 16 + 4].copy_from_slice(&amax.to_le_bytes());
                bytes
            }).collect()
        } else {
            (0..count as usize * hidden).flat_map(|at| {
                let value = bf16((((at * 31 + unit_index * 977) % 509) as f32 - 254.0) / (64.0 + unit_index as f32 * 8.0));
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            }).collect()
        };
        let mut owned = ExpertProtocolV2Request::new(500 + unit_index as u64, 17, 3, hidden as u32,
            if nvfp4_wire { ExpertV2Dtype::Nvfp4E2m1E4m3K16Amax } else { ExpertV2Dtype::Bf16 },
            (0..count).map(|row| ExpertProtocolV2RowDescriptor { row_id: u64::from(row),
                source_kind: ExpertV2SourceKind::Prefill, source_request_id: unit_index as u64,
                token_position: u64::from(row), route_offset: row * topk as u32, route_count: topk as u32 }).collect(),
            ids.iter().enumerate().map(|(route, &expert_id)| ExpertProtocolV2RouteEntry {
                row_index: (route / topk) as u32, expert_id,
                gate_weight: ((route % topk + 1 + unit_index) as f32) / 16.0 }).collect(),
            payload)?;
        owned.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        let encoded = owned.encode()?;
        let mut frame = HostAllocation::new(&library, encoded.len(), crate::shared::memory::TEST)?;
        frame.bytes_mut().copy_from_slice(&encoded);
        frames.push(frame);
        replies.push(HostAllocation::new(&library, prefix + count as usize * hidden * 2 + 64, crate::shared::memory::TEST)?);
    }
    let mut exchange = HostExpertExchange::new(unit as u32)?;
    // Submit `members` back to back (the worker stages them while nothing
    // launches), then force one flush and drain.
    let mut run = |members: &[usize], worker: &mut worker::Fp8Worker<'_>| -> Result<Vec<Vec<u8>>> {
        for &member in members {
            replies[member].bytes_mut().fill(0xa5);
            let view = ExpertProtocolV2RequestView::parse(frames[member].bytes())?;
            let request = BackboneRequest::parse(frames[member].bytes(), unit as u32)?;
            let alias = library.cuda_host_buffer_device_alias(frames[member].buffer)?;
            let mapped = ProtocolV2RequestDevicePayload {
                hidden_payload: cuteafd_ffi::CuteafdDeviceBuffer { ptr: view.hidden_payload().as_ptr().cast_mut().cast(),
                    bytes: view.hidden_payload().len(), ..alias }, execution_lane: 0,
                response_slot: Some(library.cuda_host_buffer_device_alias(replies[member].buffer)?) };
            // SAFETY: every frame and reply outlives the drain below.
            unsafe { worker.execute_mapped_payload(&request, executor, &mut exchange, mapped)?; }
        }
        worker.flush_staged()?;
        worker.synchronize()?;
        while !worker.pending_is_empty() { ensure!(worker.unit_ready()?, "drained unit not ready"); worker.complete()?; }
        Ok(members.iter().map(|&member| {
            let end = prefix + rows[member] as usize * hidden * 2;
            let bytes = replies[member].bytes();
            assert!(bytes[end..].iter().all(|&b| b == 0xa5), "reply guard overwritten");
            bytes[prefix..end].to_vec()
        }).collect())
    };
    let singles = (0..rows.len()).map(|member| run(&[member], &mut worker).map(|mut out| out.remove(0)))
        .collect::<Result<Vec<_>>>()?;
    for single in &singles {
        ensure!(single.chunks_exact(2).any(|w| w != [0, 0]), "zero singleton reply");
    }
    let flushes_before = worker.wave_flushes();
    // Pairs at 2 x unit (5504), uneven pairs, and a full unit with a tail.
    for pair in [[0usize, 1], [1, 2], [2, 3], [3, 0], [0, 4], [4, 2]] {
        let merged = run(&pair, &mut worker)?;
        for (position, &member) in pair.iter().enumerate() {
            let different = merged[position].iter().zip(&singles[member]).filter(|(a, b)| a != b).count();
            ensure!(different == 0, "members {pair:?}: member {member} differs in {different} bytes from its singleton");
        }
        eprintln!("merged wave {:?} ({} rows): byte-identical to singletons", pair,
            pair.iter().map(|&m| rows[m]).sum::<u32>());
    }
    ensure!(worker.wave_flushes() - flushes_before == 6 && worker.merged_waves() >= 6,
        "every pair must flush as one merged launch");
    drop(worker);
    let samples = std::fs::read_to_string(timing_dir.join("expert-timing-oracle-rank0.jsonl"))?;
    let samples = samples.lines().map(serde_json::from_str::<serde_json::Value>).collect::<Result<Vec<_>, _>>()?;
    let published = samples.iter().map(|sample| sample["request_id"].as_u64().unwrap_or(u64::MAX)).collect::<Vec<_>>();
    ensure!(published == [500, 501, 502, 503, 504, 500, 501, 502, 503, 500, 504],
        "expected one timing span per singleton and per merged wave, got {published:?}");
    ensure!(samples.iter().all(|sample| sample["compute_us"].as_f64().is_some_and(|us| us > 0.0)),
        "timing span without measured compute");
    std::fs::remove_dir_all(&timing_dir)?;
    eprintln!("NVFP4 merged waves ({}): {} timing spans for 5 singletons + 6 waves",
        if nvfp4_wire { "NVFP4 wire" } else { "BF16" }, samples.len());
    Ok(())
}
