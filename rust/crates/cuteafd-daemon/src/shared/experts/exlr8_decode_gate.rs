//! Real-Spark EXLR8 decode gate: every decode width 1..=64 on the serving
//! link, each row routed to distinct experts so a wave reaches the widest
//! group counts a rank sees (96+ groups below the TP6 16-row grouped switch).
//! Each rank-summed row must be finite and bit-equal to the same row sent
//! alone: the routed serving core and the 64-row grouped instance both keep
//! K1's per-row numerics, so a row that changes with its wave's width means a
//! kernel read scratch another group or another stage wrote.
use anyhow::{ensure, Result};
use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress};
use cuteafd_transport::{protocol_v2::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16,
    worker_selection::ModelAdmission, ExpertProtocolV2Request, ExpertProtocolV2RouteEntry,
    ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind, TcpTransportConfig};
use crate::shared::{memory::{DeviceAllocation, LoadStream}, placement::{spark_rings, SparkAdmission},
    spark_intake::SparkLink};
use super::tp_reference::wire_input;
use std::{path::PathBuf, sync::Arc};

const MAX_ROWS: usize = 64;

/// Row `row` of a wave routes slot `slot` to a distinct expert: a wave of `rows`
/// rows touches `min(rows * top_k, experts)` groups. `seed` varies the set per
/// layer without letting one row's experts depend on the wave width.
fn expert(geometry: ExpertGeometry, seed: usize, row: usize, slot: usize) -> u32 {
    ((seed * 31 + row * geometry.topk as usize + slot) % geometry.experts as usize) as u32
}

fn gate(slot: usize, topk: usize) -> f32 { (slot + 1) as f32 / (topk * (topk + 1) / 2) as f32 }

/// `rows` decode rows whose row `i` carries global row `first + i`'s input and routes.
fn request(geometry: ExpertGeometry, layer: usize, seed: usize, first: usize, rows: usize, id: u64)
    -> Result<ExpertProtocolV2Request> {
    let k = geometry.topk as usize;
    let dtype = ExpertV2Dtype::Fp8E4m3Ue8m0K32;
    let mut wire = Vec::with_capacity(rows * dtype.row_bytes(geometry.hidden as usize)?);
    for row in first..first + rows { wire.extend(wire_input(geometry, row, dtype)?); }
    let routes = (0..rows).flat_map(|i| (0..k).map(move |slot| ExpertProtocolV2RouteEntry {
        row_index: i as u32, expert_id: expert(geometry, seed, first + i, slot), gate_weight: gate(slot, k),
    })).collect();
    let descriptors = (0..rows).map(|i| ExpertProtocolV2RowDescriptor {
        row_id: i as u64, source_kind: ExpertV2SourceKind::Decode, source_request_id: 1,
        token_position: (first + i) as u64, route_offset: (i * k) as u32, route_count: k as u32,
    }).collect();
    let mut request = ExpertProtocolV2Request::new_bytes(id, 521, layer as u32, geometry.hidden, dtype,
        descriptors, routes, wire.into())?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

#[test]
fn decode_gate_waves_reach_the_widest_group_counts_with_width_independent_rows() -> Result<()> {
    let geometry = ExpertGeometry::GLM5;
    let k = geometry.topk as usize;
    for rows in [1, 12, 15, 16, 64] {
        let wave = request(geometry, 3, 0, 0, rows, 1)?;
        let mut experts: Vec<_> = wave.routes.iter().map(|r| r.expert_id).collect();
        experts.sort_unstable();
        experts.dedup();
        assert_eq!(experts.len(), (rows * k).min(geometry.experts as usize));
        for row in 0..rows {
            let alone = request(geometry, 3, 0, row, 1, 1)?;
            assert_eq!(&wave.routes[row * k..(row + 1) * k].iter().map(|r| (r.expert_id, r.gate_weight.to_bits()))
                .collect::<Vec<_>>(), &alone.routes.iter().map(|r| (r.expert_id, r.gate_weight.to_bits()))
                .collect::<Vec<_>>());
        }
    }
    // 12 rows give the 96 groups at which a three-unit compact activation arena overflowed.
    assert_eq!((12 * k).min(geometry.experts as usize), 96);
    Ok(())
}

#[test]
#[ignore = "requires real EXLR8 Spark workers via golden-run --spark-hosts; CUTEAFD_GOLDEN_EXLR8_MANIFEST names the manifest"]
fn native_exlr8_decode_widths_finite_and_row_invariant() -> Result<()> {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
    let snapshot = PathBuf::from(std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?);
    let manifest = std::env::var("CUTEAFD_GOLDEN_EXLR8_MANIFEST")?;
    let world: usize = std::env::var("CUTEAFD_GOLDEN_SPARK_WORLD")?.parse()?;
    let layers = std::env::var("CUTEAFD_GOLDEN_EXPERT_LAYERS")?.split(',')
        .map(str::parse).collect::<std::result::Result<Vec<usize>, _>>()?;
    let peers = std::env::var("CUTEAFD_GOLDEN_PEERS")?.split(',')
        .map(str::parse).collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(peers.len() == world && !layers.is_empty(), "decode gate world disagrees with launched peers");
    let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
    ensure!(catalog.format() == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8,
        "decode gate requires an EXLR8 checkpoint");
    let geometry = catalog.routed_experts().geometry()?;
    cuteafd_core::set_expert_geometry(geometry)
        .map_err(|_| anyhow::anyhow!("decode gate process already selected another expert geometry"))?;
    let h = geometry.hidden as usize;
    let row_bytes = ExpertV2Dtype::Bf16.row_bytes(h)?;
    // SAFETY: the official runner supplies the verified sealed native artifact.
    let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? });
    library.cuda_set_device(0)?;
    let config = TcpTransportConfig { max_frame_bytes:
        cuteafd_transport::expert::native_frame_budget(geometry, MAX_ROWS as u32)?, ..Default::default() };
    let spark_world = crate::shared::spark_topology::world(None, world)?;
    SparkAdmission::plan_manifest(&snapshot, geometry, Some(&catalog),
        &cuteafd_loader::placement::SparkWorkingSet { rtx_layers: vec![], spark_layers: layers.clone(),
            spark_ranks: world, prefill_lanes: 1 }, world, None,
        ModelAdmission { capacity: MAX_ROWS as u32, compiled_capacity: 4096,
            endpoints: vec![SparkEndpoint { ring_depth: None, capacity: MAX_ROWS as u32,
                ingress: SparkIngress::Fp8K32 }],
            rings: spark_rings(config.max_frame_bytes)? }, Some(&manifest))?.select(&peers)?;
    let zero = DeviceAllocation::new(&library, MAX_ROWS * row_bytes)?;
    let output = DeviceAllocation::new(&library, MAX_ROWS * row_bytes)?;
    library.cuda_zero_bytes(zero.buffer, zero.buffer.bytes)?;
    let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
    let mut link = SparkLink::new(&library, spark_world, &peers, MAX_ROWS as u32, config, row_bytes)?;
    link.enable_terminal_ownership(library.clone())?;
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let result = runtime.block_on(async {
        let mut id = 1;
        link.bootstrap(&request(geometry, layers[0], 0, 0, MAX_ROWS, id)?)?;
        let out = PathBuf::from(std::env::var("CUTEAFD_GOLDEN_OUT")?);
        let mut checked = 0usize;
        for (seed, &layer) in layers.iter().enumerate() {
            // Widths 1..=64 after every single-row reference: the reference
            // rows run through the routed core (one row is below every switch).
            let mut waves = Vec::with_capacity(2 * MAX_ROWS);
            for row in 0..MAX_ROWS { waves.push((row, 1)); }
            for rows in 1..=MAX_ROWS { waves.push((0, rows)); }
            let mut alone = Vec::with_capacity(MAX_ROWS);
            for (first, rows) in waves {
                id += 1;
                let request = request(geometry, layer, seed, first, rows, id)?;
                let wave = link.dispatch(&request)?;
                link.receive(wave, rows, stream.raw).await?;
                // SAFETY: full-width buffers stay owned until the stream drains;
                // receive orders every rank plane first.
                unsafe { link.reduce(zero.buffer.ptr.cast(), output.buffer.ptr.cast(), rows, stream.raw)?; }
                // SAFETY: stream and all queued storage remain live here.
                unsafe { library.cuda_stream_synchronize(stream.raw)?; }
                let mut wave = vec![0; output.buffer.bytes];
                library.copy_d2h(&mut wave, output.buffer)?;
                wave.truncate(rows * row_bytes);
                if alone.len() < MAX_ROWS {
                    ensure!(wave.chunks_exact(2).all(|v|
                        f32::from_bits((u16::from_le_bytes([v[0], v[1]]) as u32) << 16).is_finite()),
                        "EXLR8 decode layer={layer} single row {first}: non-finite partial");
                    alone.push(wave);
                    continue;
                }
                for row in 0..rows {
                    let actual = &wave[row * row_bytes..(row + 1) * row_bytes];
                    let finite = actual.chunks_exact(2).all(|v|
                        f32::from_bits((u16::from_le_bytes([v[0], v[1]]) as u32) << 16).is_finite());
                    if !finite || actual != alone[row].as_slice() {
                        std::fs::write(out.join(format!("exlr8-decode-l{layer}-m{rows}.bf16")), &wave)?;
                        anyhow::bail!("EXLR8 decode layer={layer} rows={rows} row={row}: {}",
                            if finite { "differs from the same row decoded alone" } else { "non-finite partial" });
                    }
                    checked += 1;
                }
                eprintln!("exlr8-decode world={world} layer={layer} rows={rows} groups={} ok",
                    (rows * geometry.topk as usize).min(geometry.experts as usize));
            }
        }
        std::fs::write(out.join("exlr8-decode.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "world": world, "layers": layers, "manifest": manifest, "max_rows": MAX_ROWS, "rows_checked": checked,
        }))?)?;
        Ok::<_, anyhow::Error>(())
    });
    let quiesce = link.terminal_quiesce();
    let copies = link.terminal_drain_copies();
    // SAFETY: allocations, stream and intake remain alive on every result path.
    crate::shared::decode_graph::fatal_drain(unsafe { library.cuda_stream_synchronize(stream.raw) }
        .map_err(anyhow::Error::from), "EXLR8 decode gate");
    quiesce?;
    copies?;
    link.terminal_release()?;
    result
}
