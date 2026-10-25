//! Same-quant gate through the serving engine's actual router/shared/exchange/reduce path.
use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

const REVISION: &str = "a2b599e06c59f2ccb01aaa44e6f182ba21e39f80";
const DENSE_REVISION: &str = "aca966e4e02791568aa6a4ced368624b3d897f42";
const ROWS: [usize; 5] = [1, 8, 64, 65, 128];
const SOURCES: [&str; 4] = [
    "python/reference/exlr8.py",
    "python/reference/shape_invariant.py",
    "python/tests/test_exlr8_glm_fixture.py",
    "third_party/transformers/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py",
];

fn hash_reader(mut reader: impl Read) -> Result<String> {
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn verify_files(root: &Path, hashes: &Value) -> Result<()> {
    for (name, expected) in hashes.as_object().context("fixture source hashes")? {
        ensure!(
            Path::new(name)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
            "non-relative fixture source"
        );
        ensure!(
            expected.as_str() == Some(hash_reader(std::fs::File::open(root.join(name))?)?.as_str()),
            "fixture source differs: {name}"
        );
    }
    Ok(())
}

fn fixture(
    root: &Path,
    manifest: &Value,
    name: &str,
    dtype: &str,
    shape: &[usize],
) -> Result<Vec<u8>> {
    let size = match dtype {
        "torch.uint8" => 1usize,
        "torch.bfloat16" => 2usize,
        "torch.float32" | "torch.int32" => 4,
        "torch.float64" => 8,
        _ => anyhow::bail!("fixture dtype"),
    };
    let size = shape
        .iter()
        .try_fold(size, |n, d| n.checked_mul(*d))
        .context("fixture overflow")?;
    let entry = &manifest["files"][name];
    ensure!(
        entry["dtype"] == dtype
            && entry["shape"] == serde_json::json!(shape)
            && entry["bytes"] == size,
        "fixture shape/dtype: {name}"
    );
    let bytes = std::fs::read(root.join(format!("{name}.bin")))?;
    ensure!(
        bytes.len() == size
            && entry["sha256"].as_str() == Some(format!("{:x}", Sha256::digest(&bytes)).as_str()),
        "fixture extent/hash: {name}"
    );
    Ok(bytes)
}

fn values(bytes: &[u8], size: usize) -> Result<Vec<f64>> {
    ensure!(
        matches!(size, 2 | 4 | 8) && bytes.len() % size == 0,
        "oracle extent"
    );
    let values: Vec<_> = bytes
        .chunks_exact(size)
        .map(|b| match size {
            2 => f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16) as f64,
            4 => f32::from_le_bytes(b.try_into().unwrap()) as f64,
            8 => f64::from_le_bytes(b.try_into().unwrap()),
            _ => unreachable!(),
        })
        .collect();
    ensure!(
        !values.is_empty() && values.iter().all(|x| x.is_finite()),
        "nonfinite oracle"
    );
    Ok(values)
}

fn accuracy(native: &[u8], official: &[u8], reference: &[u8], width: usize) -> Result<()> {
    let (native, official, reference) = (
        values(native, 2)?,
        values(official, 4)?,
        values(reference, 8)?,
    );
    ensure!(
        width > 0
            && native.len() == official.len()
            && native.len() == reference.len()
            && native.len() % width == 0,
        "oracle shapes"
    );
    let error = |a: &[f64], r: &[f64]| {
        a.iter()
            .zip(r)
            .map(|(a, r)| (a - r).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let check = |label: &str, a: &[f64], b: &[f64], r: &[f64]| -> Result<()> {
        let (a, b) = (error(a, r), error(b, r));
        eprintln!(
            "EXLR8 {label} native_fp64_l2={a} official_fp64_l2={b} ratio={}",
            a / b
        );
        ensure!(
            a <= 1.01 * b,
            "EXLR8 {label} exceeds same-quant FP64 error bar"
        );
        Ok(())
    };
    check("global", &native, &official, &reference)?;
    for (row, ((a, b), r)) in native
        .chunks_exact(width)
        .zip(official.chunks_exact(width))
        .zip(reference.chunks_exact(width))
        .enumerate()
    {
        check(&format!("row {row}"), a, b, r)?;
    }
    Ok(())
}

#[test]
fn gate_rejects_a_bad_row_even_when_global_error_passes() -> Result<()> {
    let native: Vec<_> = [2f32, 1000.]
        .into_iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let official: Vec<_> = [1f32, 1000.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let fp64: Vec<_> = [0f64, 0.].into_iter().flat_map(f64::to_le_bytes).collect();
    assert!(accuracy(&native, &official, &fp64, 1).is_err());
    assert!(values(&f64::NAN.to_le_bytes(), 8).is_err());
    Ok(())
}

#[test]
#[ignore = "requires pinned fixtures, native SM120 exports and real EXLR8 workers"]
fn native_layer3_same_quant() -> Result<()> {
    use clap::Parser;
    #[derive(Parser)]
    struct Args {
        #[command(flatten)]
        engine: crate::families::glm5::EngineArgs,
    }
    let root = PathBuf::from(std::env::var("CUTEAFD_GOLDEN_FIXTURES")?);
    let snapshot = PathBuf::from(std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?);
    let dense = PathBuf::from(std::env::var("CUTEAFD_EXLR8_DENSE_SNAPSHOT")?);
    let manifest: Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
    ensure!(
        manifest["schema"] == 2
            && manifest["activation_wire"] == "fp8_e4m3_ue8m0_k32_amax_floor_1e-4_routed_only"
            && manifest["seed"] == 408
            && manifest["layer"] == 3
            && manifest["rows"] == serde_json::json!(ROWS)
            && manifest["manifest"] == "up1"
            && manifest["revision"] == REVISION
            && manifest["dense_revision"] == DENSE_REVISION
            && manifest["checkpoint"] == "kindlingai/glm-5.3-exlr8-k2-k3-k4"
            && manifest["dense_checkpoint"] == "zai-org/GLM-5.3"
            && manifest["set_id"]
                == "2011c0cedde962d19fc43e74c1d409279e398d40d27a6fdb4bcb7b49ea2ea6fb"
            && manifest["error_ratio_bound"] == 1.01
            && snapshot.file_name().and_then(|x| x.to_str()) == Some(REVISION)
            && dense.file_name().and_then(|x| x.to_str()) == Some(DENSE_REVISION),
        "EXLR8 fixture identity"
    );
    let source = std::env::current_dir()?;
    ensure!(
        manifest["source_sha256"]
            .as_object()
            .context("source hashes")?
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>()
            == SOURCES.into_iter().collect(),
        "source inventory"
    );
    verify_files(&source, &manifest["source_sha256"])?;
    let expected_files: std::collections::BTreeSet<_> = [
        "config.json",
        "manifests/exlr8-manifest-up1.json",
        "experts/exlr8-layer-003.index.safetensors",
        "experts/exlr8-layer-003.k4.safetensors",
    ]
    .into_iter()
    .collect();
    ensure!(
        manifest["checkpoint_files_sha256"]
            .as_object()
            .context("checkpoint hashes")?
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>()
            == expected_files,
        "checkpoint source inventory"
    );
    verify_files(&snapshot, &manifest["checkpoint_files_sha256"])?;
    let hashes = manifest["dense_tensors_sha256"]
        .as_object()
        .context("dense source hashes")?;
    let expected_dense: std::collections::BTreeSet<_> = [
        "model.layers.3.mlp.gate.weight",
        "model.layers.3.mlp.gate.e_score_correction_bias",
        "model.layers.3.post_attention_layernorm.weight",
    ]
    .into_iter()
    .collect();
    ensure!(
        hashes
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>()
            == expected_dense,
        "dense source inventory"
    );
    let checkpoint =
        cuteafd_loader::plan::Checkpoint::open_for_role(&dense, "EXLR8 fixture identity", |n| {
            hashes.contains_key(n)
        })?;
    ensure!(
        checkpoint.tensors.len() == hashes.len(),
        "dense tensor inventory"
    );
    for tensor in &checkpoint.tensors {
        let mut file = std::fs::File::open(dense.join(&tensor.shard))?;
        file.seek(SeekFrom::Start(tensor.meta.byte_offset))?;
        ensure!(
            hashes[&tensor.meta.name].as_str()
                == Some(hash_reader(file.take(tensor.meta.byte_length))?.as_str()),
            "dense source differs: {}",
            tensor.meta.name
        );
    }
    let args = Args::try_parse_from([
        "exlr8-gate".into(),
        "--snapshot".into(),
        snapshot.into_os_string(),
        "--native-lib".into(),
        std::env::var_os("CUTEAFD_NATIVE_LIB").context("native library")?,
        "--manifest".into(),
        std::env::var_os("CUTEAFD_PROGRAM_MANIFEST").context("program manifest")?,
        "--exlr8-manifest".into(),
        "up1".into(),
        "--layers".into(),
        "4".into(),
        "--max-context".into(),
        "128".into(),
        "--pool-tokens".into(),
        "256".into(),
        "--prefill-rows".into(),
        ROWS.last().unwrap().to_string().into(),
        "--rtx-expert-layers".into(),
        "0".into(),
        "--l2-prefetch".into(),
        "off".into(),
        "--peers".into(),
        std::env::var_os("CUTEAFD_GOLDEN_PEERS").context("real worker peers")?,
    ])?
    .engine;
    let opened = crate::families::glm5::open(&args)?;
    opened.with_engine(&args, |engine, transport, runtime| {
        let transport = transport.context("native gate needs real workers")?;
        // Retain the admitted serial workspace and drain on both success and error
        // before with_engine drops transport or native owners.
        let mut workspaces = engine.lane_workspaces.borrow_mut();
        if workspaces.is_empty() {
            workspaces.push(engine.workspace(*ROWS.last().unwrap(), false)?);
        }
        let w = &workspaces[0];
        let result = (|| {
            for rows in ROWS {
                let h = engine.cfg.hidden;
                let load = |suffix: &str, dtype: &str, shape: &[usize]| {
                    fixture(&root, &manifest, &format!("r{rows}_{suffix}"), dtype, shape)
                };
                engine.put(&w.h, &load("residual", "torch.bfloat16", &[rows, h])?)?;
                engine.run(
                    "glm_norm",
                    &[
                        ("residual", w.h.buffer.ptr),
                        ("delta0", w.delta.buffer.ptr),
                        ("delta1", w.delta.buffer.ptr),
                        ("weight", engine.weights.layers[3].ptr("post_norm")?),
                        ("out", w.x.buffer.ptr),
                    ],
                    &[Scalar::I32(rows as i32), Scalar::I32(0)],
                )?;
                engine.moe(
                    w,
                    3,
                    &engine.weights.layers[3],
                    rows,
                    "m4096",
                    false,
                    transport,
                    runtime,
                )?;
                let staged = w.router_host.borrow();
                let request = expert_request(
                    staged.bytes(),
                    3,
                    rows,
                    h,
                    engine.cfg.topk,
                    false,
                    engine.spark_bf16,
                )?;
                ensure!(
                    request.header.hidden_dtype
                        == cuteafd_transport::ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                    "same-quant gate requires standard Fp8K32 requests"
                );
                let wire = request.hidden_payload.as_ref();
                let out = PathBuf::from(std::env::var("CUTEAFD_GOLDEN_OUT")?);
                std::fs::write(out.join(format!("r{rows}_actual_wire.bin")), wire)?;
                let expected_wire = load("wire", "torch.uint8", &[rows, h + h / 32])?;
                ensure!(
                    wire == expected_wire,
                    "native request differs from NumPy FP8 K32 oracle at {rows} rows"
                );
                drop(staged);
                // The ordinary reducer writes delta; norm applies the residual in place.
                engine.run(
                    "glm_norm",
                    &[
                        ("residual", w.h.buffer.ptr),
                        ("delta0", w.delta.buffer.ptr),
                        ("delta1", w.delta.buffer.ptr),
                        ("weight", engine.weights.layers[3].ptr("post_norm")?),
                        ("out", w.x.buffer.ptr),
                    ],
                    &[Scalar::I32(rows as i32), Scalar::I32(1)],
                )?;
                let actual = engine.download(&w.h, rows * h * 2)?;
                let ids = engine.download(&w.route_ids, rows * engine.cfg.topk * 4)?;
                let expected = load("ids", "torch.int32", &[rows, engine.cfg.topk])?;
                for (actual, expected) in ids
                    .chunks_exact(engine.cfg.topk * 4)
                    .zip(expected.chunks_exact(engine.cfg.topk * 4))
                {
                    let sort = |bytes: &[u8]| {
                        let mut ids: Vec<_> = bytes
                            .chunks_exact(4)
                            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                            .collect();
                        ids.sort_unstable();
                        ids
                    };
                    ensure!(
                        sort(actual) == sort(expected),
                        "native router selected different experts"
                    );
                }
                accuracy(
                    &actual,
                    &load("official_output", "torch.float32", &[rows, h])?,
                    &load("fp64_output", "torch.float64", &[rows, h])?,
                    h,
                )?;
            }
            Ok(())
        })();
        // SAFETY: all engine, workspace and transport allocations are retained here.
        let drained = unsafe { engine.library.cuda_stream_synchronize(engine.stream) };
        result.and(drained.map_err(Into::into))
    })
}

/// Keep projection and attention row shapes identical: the serial arm finishes
/// each chunk, while the pipeline schedules the same chunks layer-major.
#[test]
#[ignore = "requires native SM120 exports, two GPUs and real EXLR8 workers"]
fn native_prefill_lanes_byte_exact() -> Result<()> {
    prefill_lanes_byte_exact(3072, 1024, 3)
}

#[test]
#[ignore = "requires native SM120 exports, two GPUs and real EXLR8 workers"]
fn native_prefill_two_waves_byte_exact() -> Result<()> {
    ensure!(std::env::var("CUTEAFD_SPARK_INTAKE").as_deref() == Ok("pinned")
        && std::env::var("CUTEAFD_SPARK_EGRESS").as_deref() == Ok("copied"),
        "two-wave exactness gate requires pinned intake and copied egress on both links");
    prefill_lanes_byte_exact(8192, 4096, 2)
}

#[test]
#[ignore = "requires native SM120 exports, two GPUs and real EXLR8 workers"]
fn native_prefill_lanes_long_byte_exact() -> Result<()> {
    let lengths = std::env::var("CUTEAFD_EXLR8_LANE_LENGTHS").unwrap_or_else(|_| "10771,16384,32768".into());
    for length in lengths.split(',') {
        prefill_lanes_byte_exact(length.parse()?, 4096, 3)?;
    }
    Ok(())
}

fn prefill_lanes_byte_exact(total: usize, chunk: usize, waves: usize) -> Result<()> {
    use clap::Parser;
    #[derive(Parser)]
    struct Args {
        #[command(flatten)]
        engine: crate::families::glm5::EngineArgs,
    }
    let context = total.next_multiple_of(2 * PAGE_ROWS);
    let mut shape = Vec::new();
    let mut start = 0;
    while start < total {
        let batch = (total - start).min(chunk * waves);
        let count = waves.min(batch / MIN_LANE_ROWS).max(1);
        let per_lane = batch.div_ceil(count).next_multiple_of(PAGE_ROWS);
        let mut first = 0;
        while first < batch {
            let rows = per_lane.min(batch - first);
            shape.push((start + first, rows));
            first += rows;
        }
        start += batch;
    }
    eprintln!("EXLR8 lane golden total={total} chunk={chunk} waves={waves} shape={shape:?}");
    let args = Args::try_parse_from([
        "exlr8-lanes".into(),
        "--snapshot".into(),
        std::env::var_os("CUTEAFD_GOLDEN_SNAPSHOT").context("native snapshot")?,
        "--native-lib".into(),
        std::env::var_os("CUTEAFD_NATIVE_LIB").context("native library")?,
        "--manifest".into(),
        std::env::var_os("CUTEAFD_PROGRAM_MANIFEST").context("program manifest")?,
        "--exlr8-manifest".into(), "up1".into(),
        "--device".into(), "0".into(),
        "--split-device".into(), "1".into(),
        "--attention-placement".into(), "context".into(),
        "--max-context".into(), context.to_string().into(),
        "--pool-tokens".into(), context.to_string().into(),
        "--prefill-rows".into(), chunk.to_string().into(),
        "--prefill-waves".into(), waves.to_string().into(),
        "--rtx-expert-layers".into(), "0".into(),
        "--l2-prefetch".into(), "off".into(),
        "--peers".into(),
        std::env::var_os("CUTEAFD_GOLDEN_PEERS").context("real worker peers")?,
    ])?.engine;
    let opened = crate::families::glm5::open(&args)?;
    opened.with_engine(&args, |engine, transport, runtime| {
        ensure!(engine.prefill_lanes == waves
            && engine.prefill_link.borrow().as_ref().is_some_and(|link| link.waves() == waves),
            "lane exactness gate requires exactly {waves} admitted waves on the prefill link");
        ensure!(engine.native_experts.len() == 2, "lane gate requires native shared owners");
        let transport = transport.context("lane gate needs real workers")?;
        let mut allocator = PageAllocator::new(engine);
        let placement = allocator.admit(total)?;
        let tokens: Vec<u32> = (0..total)
            .map(|i| (17 + i * 7919) as u32 % engine.cfg.vocab_size as u32).collect();
        let tables = shape.iter().map(|&(start, rows)|
            engine.prefill_tables(&placement, start, rows)).collect::<Result<Vec<_>>>()?;
        let chunks: Vec<_> = shape.iter().map(|&(start, rows)| &tokens[start..start + rows]).collect();
        let caches = |rank| -> Result<(&Vec<Dev<'_>>, &Vec<Option<Dev<'_>>>)> {
            if rank == 0 { Ok((&engine.kv, &engine.index)) }
            else { let peer = engine.peer()?; Ok((&peer.kv, &peer.index)) }
        };
        let drain = || -> Result<()> {
            for rank in 0..engine.ranks() {
                // SAFETY: both engines' streams and all test operands remain live.
                engine.on(rank, || unsafe {
                    engine.library.cuda_stream_synchronize(engine.stream_of(rank))
                })?;
            }
            Ok(())
        };
        let clear = || -> Result<()> {
            drain()?;
            for rank in 0..engine.ranks() {
                let (kv, index) = caches(rank)?;
                engine.on(rank, || {
                    for buffer in kv.iter().chain(index.iter().flatten()) {
                        engine.library.cuda_zero_bytes(buffer.buffer, buffer.buffer.bytes)?;
                    }
                    Ok(())
                })?;
            }
            Ok(())
        };
        let snapshot = || -> Result<Vec<Vec<Vec<u8>>>> {
            drain()?;
            (0..engine.ranks()).map(|rank| {
                let (kv, index) = caches(rank)?;
                engine.on(rank, || kv.iter().zip(index).map(|(kv, index)| {
                    let mut bytes = vec![0; kv.buffer.bytes];
                    engine.library.copy_d2h(&mut bytes, kv.buffer)?;
                    if let Some(index) = index {
                        let first = bytes.len();
                        bytes.resize(first + index.buffer.bytes, 0);
                        engine.library.copy_d2h(&mut bytes[first..], index.buffer)?;
                    }
                    Ok(bytes)
                }).collect::<Result<Vec<_>>>())
            }).collect()
        };
        let result = (|| {
            clear()?;
            let mut serial = None;
            for (lane, (tables, tokens)) in tables.iter().zip(&chunks).enumerate() {
                eprintln!("EXLR8 lane golden serial total={total} unit={lane} rows={}", tokens.len());
                let logits = engine.step(tables, tokens, usize::from(lane + 1 == chunks.len()),
                    Some((&mut *transport, runtime)), None)?;
                if lane + 1 == chunks.len() {
                    serial = Some(logits.context("serial final logits")?.to_host(engine.library)?);
                } else {
                    ensure!(logits.is_none(), "zero-row serial prefill returned logits");
                }
            }
            let serial_state = snapshot()?;
            clear()?;
            let mut pipelined = None;
            for (batch, (tables, chunks)) in tables.chunks(waves).zip(chunks.chunks(waves)).enumerate() {
                let last = (batch + 1) * waves >= shape.len();
                eprintln!("EXLR8 lane golden pipeline total={total} batch={batch} rows={:?}",
                    chunks.iter().map(|chunk| chunk.len()).collect::<Vec<_>>());
                let mut link = engine.prefill_link.borrow_mut();
                let link = link.as_mut().context("EXLR8 lane golden needs the prefill link")?;
                let runtime = engine.prefill_runtime.as_ref().context("prefill runtime")?;
                let logits = engine.host_logits(engine.step_lanes(tables, chunks, usize::from(last), link, runtime)?)?;
                if last { pipelined = logits; }
                else { ensure!(logits.is_none(), "zero-row pipeline prefill returned logits"); }
            }
            let pipelined = pipelined.context("pipelined final logits")?;
            let pipelined_state = snapshot()?;
            for (rank, (serial, pipelined)) in serial_state.iter().zip(&pipelined_state).enumerate() {
                for (layer, (serial, pipelined)) in serial.iter().zip(pipelined).enumerate() {
                    ensure!(serial == pipelined, "KV/index bytes differ at rank {rank} layer {layer}");
                }
            }
            let serial = serial.context("serial logits")?;
            ensure!(serial.len() == pipelined.len() && serial.iter().zip(&pipelined)
                .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits()),
                "fixed-chunk prefill final logits differ");
            eprintln!("EXLR8 lanes exact: total={total} chunk={chunk} waves={waves} ranks={} layers={} logits={}",
                engine.ranks(), engine.weights.layers.len(), serial.len());
            Ok(())
        })();
        let drained = drain();
        allocator.release(placement);
        result.and(drained)
    })
}
