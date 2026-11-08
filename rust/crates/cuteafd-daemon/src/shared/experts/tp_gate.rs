//! Real-Spark source-native gate through the serving handshake and rank reducer.
use anyhow::{ensure, Result};
use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress};
use cuteafd_loader::formats::fp8_experts::ExpertFormat;
use cuteafd_transport::{expert::SparkWorld, worker_selection::ModelAdmission,
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
    ExpertV2Dtype, ExpertV2SourceKind, TcpTransportConfig,
    protocol_v2::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};
use crate::shared::{memory::{DeviceAllocation, LoadStream}, placement::{spark_rings, SparkAdmission},
    spark_intake::SparkLink};
use super::tp_reference::{error, expert, outputs, reference, route_weight, wire_input};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};
fn checkpoint_ingress(snapshot: &std::path::Path) -> Result<(SparkIngress, ExpertV2Dtype)> {
    let checkpoint = cuteafd_loader::plan::Checkpoint::inventory(snapshot)?;
    Ok(if checkpoint.spark_input_is_bf16() {
        (SparkIngress::Bf16, ExpertV2Dtype::Bf16)
    } else { (SparkIngress::Fp8K32, ExpertV2Dtype::Fp8E4m3Ue8m0K32) })
}

fn partial_row_bytes(geometry: ExpertGeometry) -> Result<usize> {
    // Rank-local expert reduction returns one hidden-wide BF16 row, regardless of ingress.
    ExpertV2Dtype::Bf16.row_bytes(geometry.hidden as usize)
}

fn request(geometry: ExpertGeometry, layer: usize, rows: usize, dtype: ExpertV2Dtype)
    -> Result<ExpertProtocolV2Request> {
    let k = geometry.topk as usize;
    let pitch = dtype.row_bytes(geometry.hidden as usize)?;
    let mut wire = Vec::with_capacity(rows * pitch);
    for row in 0..rows { wire.extend(wire_input(geometry, row, dtype)?); }
    let routes = (0..rows).flat_map(|row| (0..k).map(move |slot| ExpertProtocolV2RouteEntry {
        row_index: row as u32, expert_id: expert(geometry, layer, row, slot) as u32,
        gate_weight: route_weight(geometry, slot),
    })).collect();
    let descriptors = (0..rows).map(|row| ExpertProtocolV2RowDescriptor {
        row_id: row as u64, source_kind: ExpertV2SourceKind::Prefill,
        source_request_id: 1, token_position: row as u64, route_offset: (row * k) as u32,
        route_count: k as u32,
    }).collect();
    let mut request = ExpertProtocolV2Request::new_bytes(layer as u64 + rows as u64, 521,
        layer as u32, geometry.hidden, dtype, descriptors, routes, wire.into())?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

/// A gate drives one worker at `rows`: one decode and one prefill owner at
/// that extent over the checkpoint backend's export ladder.
pub(super) fn gate_executions(catalog: &cuteafd_loader::OfficialV41Catalog, rows: u32)
    -> Result<cuteafd_core::SparkExecutions> {
    let backend = cuteafd_loader::plan::spark_executions::SparkBackend::for_catalog(catalog)?;
    Ok(cuteafd_core::SparkExecutions::admit(&[rows], rows, rows, 1, &backend.exports())?)
}

fn validate_source(catalog: &cuteafd_loader::OfficialV41Catalog, family: &str, format: &str)
    -> Result<ExpertGeometry> {
    let geometry = catalog.routed_experts().geometry()?;
    ensure!(geometry.family() == Some(family), "rank-sum family {family:?} disagrees with checkpoint {geometry:?}");
    let actual = if catalog.exl3().is_some() { "exl3" }
        else if let Some(tensors) = catalog.fp8() {
            match tensors.format() {
                ExpertFormat::Mxfp4 => "mxfp4", ExpertFormat::Fp8Block128 => "fp8",
                other => anyhow::bail!("rank-sum source decoder not qualified for {other:?}"),
            }
        } else {
            ensure!(geometry.family() == Some("v41") && catalog.nvfp4().is_none(),
                "rank-sum source decoder requires native V4.1, FP8, MXFP4 or EXL3");
            "mxfp4"
        };
    ensure!(actual == format, "rank-sum format {format:?} disagrees with checkpoint {actual}");
    ensure!(matches!(family, "v41" | "glm" | "mimo" | "mimop"),
        "rank-sum activation oracle is not qualified for family {family:?}");
    Ok(geometry)
}

fn checkpoint_world(catalog: &cuteafd_loader::OfficialV41Catalog, ranks: usize) -> Result<SparkWorld> {
    let world = crate::shared::spark_topology::world(ranks)?;
    crate::shared::spark_topology::require_wire(world, catalog)?;
    Ok(world)
}

#[test]
fn checkpoint_worlds_are_canonical_for_every_format() -> Result<()> {
    use cuteafd_loader::plan::testing as fixture;
    let root = tempfile::tempdir()?;
    fixture::write_v41_snapshot(root.path());
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    for ranks in [3, 4, 6] {
        assert_eq!(checkpoint_world(&catalog, ranks)?, SparkWorld::canonical(ranks)?);
    }
    let root = tempfile::tempdir()?;
    fixture::write_v4_pro_exl3_snapshot(root.path());
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    assert_eq!(checkpoint_world(&catalog, 3)?, SparkWorld::canonical(3)?);
    Ok(())
}

#[test]
fn requests_use_checkpoint_geometry_and_two_repeated_oracle_rows() -> Result<()> {
    for geometry in [ExpertGeometry::DEEPSEEK_V41, ExpertGeometry::GLM5,
        ExpertGeometry::MIMO_V2_FLASH, ExpertGeometry::MIMO_V26_PRO] {
        for dtype in [ExpertV2Dtype::Bf16, ExpertV2Dtype::Fp8E4m3Ue8m0K32] {
            let small = request(geometry, 3, 1, dtype)?;
            let wide = request(geometry, 3, 16, dtype)?;
            let pitch = dtype.row_bytes(geometry.hidden as usize)?;
            assert_eq!(small.header.hidden_dim, geometry.hidden);
            assert_eq!(wide.header.hidden_dim, geometry.hidden);
            assert_eq!(small.header.hidden_dtype, dtype);
            assert_eq!(wide.header.hidden_row_stride_bytes as usize, pitch);
            assert_eq!(small.hidden_payload.len(), pitch);
            assert_eq!(wide.hidden_payload.len(), 16 * pitch);
            assert_eq!(&wide.hidden_payload[..pitch], &small.hidden_payload[..]);
            for row in 2..16 {
                assert_eq!(&wide.hidden_payload[row * pitch..(row + 1) * pitch],
                    &wide.hidden_payload[(row % 2) * pitch..(row % 2 + 1) * pitch]);
            }
        }
    }
    Ok(())
}


#[test]
fn compact_partial_planes_use_bf16_rows_independently_of_ingress_and_topk() -> Result<()> {
    for geometry in [ExpertGeometry::DEEPSEEK_V41, ExpertGeometry::GLM5,
        ExpertGeometry::MIMO_V2_FLASH, ExpertGeometry::MIMO_V26_PRO] {
        let output_pitch = partial_row_bytes(geometry)?;
        assert_eq!(output_pitch, geometry.hidden as usize * 2);
        for dtype in [ExpertV2Dtype::Bf16, ExpertV2Dtype::Fp8E4m3Ue8m0K32] {
            for rows in [1, 16] {
                let request = request(geometry, 3, rows, dtype)?;
                assert_eq!(request.header.row_count as usize, rows);
                assert_eq!(request.routes.len(), rows * geometry.topk as usize);
                assert_ne!(request.header.flags & EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16, 0);
                let plane_bytes = rows * output_pitch;
                assert_eq!(plane_bytes, rows * geometry.hidden as usize * 2);
                if dtype == ExpertV2Dtype::Fp8E4m3Ue8m0K32 {
                    assert!(request.hidden_payload.len() < plane_bytes);
                } else {
                    assert_eq!(request.hidden_payload.len(), plane_bytes);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn gate_ingress_matches_production_checkpoint_contract() -> Result<()> {
    use cuteafd_loader::plan::testing as fixture;
    for (write, ingress, dtype) in [
        (fixture::write_v41_snapshot as fn(&std::path::Path), SparkIngress::Fp8K32, ExpertV2Dtype::Fp8E4m3Ue8m0K32),
        (fixture::write_v41_nvfp4_snapshot, SparkIngress::Bf16, ExpertV2Dtype::Bf16),
    ] {
        let root = tempfile::tempdir()?;
        write(root.path());
        assert_eq!(checkpoint_ingress(root.path())?, (ingress, dtype));
    }
    let root = tempfile::tempdir()?;
    fixture::write_v4_pro_exl3_snapshot(root.path());
    assert_eq!(checkpoint_ingress(root.path())?, (SparkIngress::Fp8K32, ExpertV2Dtype::Fp8E4m3Ue8m0K32));
    Ok(())
}

#[test]
#[ignore = "requires sealed packages and real Spark workers via golden-run; family, format and world are explicit"]
fn native_tp_rank_sum() -> Result<()> {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
    let snapshot = PathBuf::from(std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?);
    let family = std::env::var("CUTEAFD_GOLDEN_FAMILY")?;
    let format = std::env::var("CUTEAFD_GOLDEN_FORMAT")?;
    let world: usize = std::env::var("CUTEAFD_GOLDEN_SPARK_WORLD")?.parse()?;
    ensure!((1..=cuteafd_core::MAX_SPARK_WORLD).contains(&world), "rank-sum requires 1..8 ranks; checkpoint recipe validates widths");
    let layers = std::env::var("CUTEAFD_GOLDEN_EXPERT_LAYERS")?.split(',')
        .map(str::parse).collect::<std::result::Result<Vec<usize>, _>>()?;
    ensure!(layers.len() >= world && layers.windows(2).all(|w| w[1] == w[0] + 1),
        "rank-sum needs at least one full world of consecutive absolute routed layers");
    let peers = std::env::var("CUTEAFD_GOLDEN_PEERS")?.split(',')
        .map(str::parse).collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(peers.len() == world, "rank-sum world disagrees with launched peers");
    let view = cuteafd_loader::plan::checkpoint::CheckpointView::select(&snapshot,
        cuteafd_loader::plan::checkpoint::AttentionMode::Hqa)?;
    let catalog = cuteafd_loader::read_expert_catalog_view(&view)?;
    let geometry = validate_source(&catalog, &family, &format)?;
    let (ingress, dtype) = checkpoint_ingress(&snapshot)?;
    let output_row_bytes = partial_row_bytes(geometry)?;
    let spark_world = checkpoint_world(&catalog, world)?;
    cuteafd_core::set_expert_geometry(geometry)
        .map_err(|_| anyhow::anyhow!("rank-sum process already selected another expert geometry"))?;
    ensure!(layers.iter().all(|&l| l >= catalog.routed_experts().first_layer && l < geometry.layers as usize),
        "rank-sum layer is outside the checkpoint's routed set");
    let columns = outputs(geometry);
    let mut references = BTreeMap::new();
    for &layer in &layers { references.insert(layer, reference(&catalog, geometry, layer, &columns, dtype)?); }
    // SAFETY: the official runner supplies the verified sealed native artifact.
    let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? });
    library.cuda_set_device(0)?;
    crate::shared::inventory::preload_native_modules(&library)?;
    let config = TcpTransportConfig::default();
    SparkAdmission::plan_view(&view, geometry, Some(&catalog),
        &cuteafd_loader::placement::SparkWorkingSet { rtx_layers: vec![], spark_layers: layers.clone(),
            spark_ranks: world, prefill_lanes: 1 }, world,
        ModelAdmission::new(vec![SparkEndpoint { ring_depth: None, capacity: 16, ingress, partial_reply_bytes: None }],
            spark_rings(config.max_frame_bytes)?, gate_executions(&catalog, 16)?)?, None)?.select(&peers)?;
    let h = geometry.hidden as usize;
    let zero = DeviceAllocation::new(&library, 16 * output_row_bytes, crate::shared::memory::TEST)?;
    let output = DeviceAllocation::new(&library, 16 * output_row_bytes, crate::shared::memory::TEST)?;
    library.cuda_zero_bytes(zero.buffer, zero.buffer.bytes)?;
    let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
    let mut link = SparkLink::new(&library, spark_world, &peers, 16, config, output_row_bytes)?;
    link.enable_terminal_ownership(library.clone())?;
    let result = (|| -> Result<()> {
        link.bootstrap(&cuteafd_transport::SparkSessionShape::from_request(&request(geometry, layers[0], 16, dtype)?)?)?;
        let out = PathBuf::from(std::env::var("CUTEAFD_GOLDEN_OUT")?);
        let mut metrics = Vec::new();
        for &layer in &layers {
            for rows in [1, 16] {
                let request = request(geometry, layer, rows, dtype)?;
                let exchange = link.exchange()?;
                let payload_bytes = exchange.compact_payload(request.header.row_count as usize);
                let wave = exchange.send_request(request, payload_bytes)?;
                let slot = wave.slot();
                exchange.land(wave, stream.raw)?;
                // SAFETY: full-width buffers and rank planes stay owned until
                // the stream drains; `land` orders every rank plane first.
                unsafe { exchange.reduce(slot, zero.buffer.ptr.cast(), output.buffer.ptr.cast(), rows, stream.raw)?; }
                // SAFETY: stream and all queued storage remain live here.
                unsafe { library.cuda_stream_synchronize(stream.raw)?; }
                let mut bytes = vec![0; output.buffer.bytes];
                library.copy_d2h(&mut bytes, output.buffer)?;
                let mut actual = Vec::new();
                let mut expected = Vec::new();
                for row in 0..rows {
                    for (sample, &col) in columns.iter().enumerate() {
                        let index = (row * h + col) * 2;
                        actual.push(f32::from_bits((u16::from_le_bytes([bytes[index], bytes[index + 1]]) as u32) << 16) as f64);
                        expected.push(references[&layer][row % 2][sample]);
                    }
                }
                let (relative, cosine) = error(&actual, &expected)?;
                ensure!(relative <= 0.05 && cosine >= 0.999,
                    "rank-sum layer={layer} rows={rows}: relative L2={relative} cosine={cosine}");
                eprintln!("rank-sum family={family} format={format} world={world} layer={layer} rows={rows} relative_l2={relative:.6} cosine={cosine:.8}");
                std::fs::write(out.join(format!("rank-sum-l{layer}-m{rows}.bf16")), &bytes[..rows * h * 2])?;
                metrics.push(serde_json::json!({"family":family,"format":format,"world":world,
                    "layer":layer,"rows":rows,"columns":columns,"relative_l2":relative,"cosine":cosine}));
            }
        }
        std::fs::write(out.join("rank-sum.json"), serde_json::to_vec_pretty(&metrics)?)?;
        crate::shared::spark_waves::healthy()
    })();
    let quiesce = link.terminal_quiesce();
    let copies = link.terminal_drain_copies();
    // SAFETY: allocations, stream and intake remain alive on every result path.
    crate::shared::decode_graph::fatal_drain(unsafe { library.cuda_stream_synchronize(stream.raw) }
        .map_err(anyhow::Error::from), "shared rank-sum gate");
    quiesce?;
    copies?;
    link.terminal_release()?;
    result
}
