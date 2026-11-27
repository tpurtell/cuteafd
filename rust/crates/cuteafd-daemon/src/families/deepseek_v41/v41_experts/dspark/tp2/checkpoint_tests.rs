use super::{pair::Pair, rank::Weights};
use crate::families::deepseek_v41::v41_experts::{ExpertLayer, ExpertWeights};
use crate::shared::memory::device::{Device, Stream};
use anyhow::{ensure, Result};
use cuteafd_ffi::NativeLibrary;

fn bf16(v: f32) -> u16 {
    let b = v.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}
fn values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|v| f32::from_bits(u32::from(u16::from_ne_bytes([v[0], v[1]])) << 16))
        .collect()
}

/// Published CPU allocation manifests must agree with the serving owners,
/// including both ranks' fork/join scratch and sequential weight-load peaks.
#[test]
#[ignore = "requires native AOT, checkpoint and two CUDA GPUs; metadata only"]
fn cuda_dspark_tp2_planner_matches_serving_inventory() -> Result<()> {
    use cuteafd_loader::placement::families::deepseek_v41::{owners, resident};
    let path = std::path::PathBuf::from(std::env::var("CUTEAFD_NATIVE_LIB")?);
    // SAFETY: the explicitly supplied WIP native library is trusted.
    let lib = unsafe { NativeLibrary::load(&path)? };
    let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(
        &std::env::var("CUTEAFD_SNAPSHOT")?))?;
    let config = serde_json::from_slice(&std::fs::read(catalog.snapshot().join("config.json"))?)?;
    let fp8 = owners::native_manifest(&path, "V41_FP8_AOT.json")?;
    let full = owners::native_manifest(&path, "V41_EXPERT_AOT.json")?;
    let halves = owners::native_manifest(&path, "V41_EXPERT_WORKSPACES/mxfp4_dspark_tp2.json")?;
    let programs = owners::NativePrograms::new(&fp8, u32::try_from(lib.sm_count()?)?)?;
    let weights = resident::resident_headers(&config,
        catalog.tensors().iter().map(|t| &t.metadata), 2, true)
        .ok_or_else(|| anyhow::anyhow!("unsupported draft resident headers"))?;
    let all = weights.iter().flatten().filter(|i| i.category == cuteafd_core::memory_layout::Category::Drafter)
        .map(|i| i.bytes).sum::<u64>();
    let half = 3 * resident::native_expert_bytes(128, 5120, 1152);
    let home = all - 3 * resident::native_expert_bytes(128, 5120, 2304) + half;
    for requests in [1, 2, 16, 32, 64] {
        for width in [5, 7] {
            let lanes = if requests == 1 { 1 } else { 2 };
            let actual = super::DsparkWeights::distributed_inventory(&lib, &catalog, 256,
                cuteafd_core::V41_VERIFY_ROWS as u32, requests, 2, width, true, 64640, 0)?;
            let planned = programs.dspark_tp2(&full, &halves, 256, owners::DECODE_LANE_ROWS, u64::from(requests), lanes,
                width as u64, resident::Fp8Head::configured())?;
            for demand in &actual.runtime {
                let expected = planned.iter().find(|d| d.gpu == demand.gpu && d.group == demand.group)
                    .ok_or_else(|| anyhow::anyhow!("missing planner owner {}", demand.group))?;
                assert_eq!(demand.bytes, expected.bytes,
                    "C{requests} width={width} GPU{} {}", demand.gpu, demand.group);
            }
            assert_eq!(actual.runtime.len(), planned.len());
            assert_eq!((actual.resident, actual.peer_half), (home, half));
            assert_eq!(actual.loading_experts, [half; 2]);
            for (index, rank) in [1, 0].into_iter().enumerate() {
                let stage = catalog.expert_staging(cuteafd_loader::V41ExpertSelection::DsparkTp2 {
                    stage: 0, expert: 0, rank })?.staging_bytes() as u64;
                assert!(actual.staging[index] >= stage);
                assert_eq!(actual.loader_peak[index], (half + actual.staging[index])
                    .max(if index == 0 { home } else { half }));
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires checkpoint, both RTX GPUs and about 7 GiB free on RTX1"]
fn cuda_dspark_tp2_checkpoint_experts_and_graph() -> Result<()> {
    let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
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
    let owner = devices[1];
    let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?),
    )?;
    let weights = Weights::load_pair(devices, &catalog, [16usize << 30; 2], super::LAYOUT)?;
    eprintln!(
        "draft TP2 resident bytes: {:?}",
        weights.each_ref().map(|w| w.resident_bytes())
    );
    let mut pair = Pair::new(weights, 256, super::LAYOUT)?;
    let root = Stream::new(owner)?;
    for stage in 0..3 {
        let full = owner.own(|| {
            ExpertWeights::load(&lib, &catalog, ExpertLayer::Dspark { stage }, 8usize << 30)
        })?;
        let mut reference = owner.own(|| full.execution(256, 256, 2usize << 30))?;
        for (rows, include_shared) in [1u32, 7, 16, 40, 65, 112, 1]
            .into_iter()
            .flat_map(|rows| [false, true].map(|shared| (rows, shared)))
        {
            let inputs = reference.inputs();
            let shared = reference.shared().unwrap();
            // Warm first so capture only encounters already initialized modules.
            for input in inputs {
                owner.run(|| lib.copy_h2d(input, &vec![0; input.bytes]))?;
            }
            owner.run(|| lib.copy_h2d(shared, &vec![0; shared.bytes]))?;
            unsafe {
                pair.enqueue(
                    stage,
                    rows,
                    inputs,
                    include_shared.then_some(shared),
                    root.raw,
                )?;
            }
            root.drain()?;
            pair.drain_peer()?;
            let graph = owner.run(|| unsafe {
                lib.cuda_graph_begin_capture(root.raw)?;
                let queued = pair.enqueue(
                    stage,
                    rows,
                    inputs,
                    include_shared.then_some(shared),
                    root.raw,
                );
                let captured = lib.cuda_graph_end_capture_for_test(root.raw);
                match (queued, captured) {
                    (Ok(()), Ok(graph)) => Ok(graph),
                    (Err(e), Ok(graph)) => {
                        lib.cuda_graph_exec_destroy(graph)?;
                        Err(e)
                    }
                    (Err(e), Err(_)) | (Ok(()), Err(e)) => Err(e),
                }
            })?;
            let checked = (|| -> Result<()> {
                for seed in [1usize, 19, 83] {
                    let hidden: Vec<u8> = (0..rows as usize * 5120)
                        .flat_map(|i| {
                            bf16((((i * 37 + seed * 11) % 257) as f32 - 128.0) * 0.003)
                                .to_ne_bytes()
                        })
                        .collect();
                    let ids: Vec<u8> = (0..rows as usize * 3)
                        .flat_map(|i| {
                            (((i / 3 * 7 + i % 3 * 41 + seed) % 128) as i32).to_ne_bytes()
                        })
                        .collect();
                    let routing: Vec<u8> = (0..rows as usize * 3)
                        .flat_map(|i| (0.15 + (i % 3) as f32 * 0.1).to_ne_bytes())
                        .collect();
                    let shared_data: Vec<u8> = (0..rows as usize * 5120)
                        .flat_map(|i| bf16(((i + seed) % 17) as f32 * 0.001).to_ne_bytes())
                        .collect();
                    for (buffer, data) in inputs.into_iter().zip([hidden, ids, routing]) {
                        owner.run(|| lib.copy_h2d(buffer, &data))?;
                    }
                    owner.run(|| lib.copy_h2d(shared, &shared_data))?;
                    owner.run(|| unsafe {
                        reference.launch(rows, include_shared)?;
                        reference.synchronize()
                    })?;
                    // Compare capture/replay with the identical eager TP2 route
                    // before comparing against the unsplit checkpoint kernel.
                    owner.run(|| unsafe {
                        pair.enqueue(stage, rows, inputs, include_shared.then_some(shared), root.raw)
                    })?;
                    root.drain()?;
                    let mut eager = vec![0; rows as usize * 10240];
                    root.copy_d2h(&mut eager, pair.output())?;
                    owner.run(|| unsafe { lib.cuda_graph_launch(graph, root.raw) })?;
                    root.drain()?;
                    ensure!(unsafe { lib.cuda_stream_query(pair.peer_stream().raw)? },
                        "root completion did not cover the producer's route write");
                    let mut actual = vec![0; rows as usize * 10240];
                    let mut expected = actual.clone();
                    root.copy_d2h(&mut actual, pair.output())?;
                    assert_eq!(actual, eager, "draft TP2 eager/replay bytes differ at stage={stage} rows={rows} shared={include_shared} seed={seed}");
                    owner.run(|| lib.copy_d2h_quiescent(&mut expected, reference.output().unwrap()))?;
                    let (actual, expected) = (values(&actual), values(&expected));
                    let mut error = 0f64;
                    let mut norm = 0f64;
                    let mut max = 0f32;
                    let mut exact = 0;
                    for (&a, &b) in actual.iter().zip(&expected) {
                        ensure!(a.is_finite() && b.is_finite(), "nonfinite draft output");
                        let d = (a - b).abs();
                        max = max.max(d);
                        error += f64::from(d).powi(2);
                        norm += f64::from(b).powi(2);
                        exact += usize::from(a == b);
                    }
                    let relative = (error / norm.max(1e-30)).sqrt();
                    eprintln!("draft TP2 stage={stage} rows={rows} shared={include_shared} seed={seed} relative_rms={relative:.7} max_abs={max:.7} exact={exact}/{}",actual.len());
                    ensure!(
                        relative < 0.0001,
                        "draft TP2 differs from full expert: relative RMS {relative}"
                    );
                }
                Ok(())
            })();
            let drained = root.drain().and(pair.drain_peer());
            owner.run(|| unsafe { lib.cuda_graph_exec_destroy(graph) })?;
            drained?;
            checked?;
        }
    }
    Ok(())
}
