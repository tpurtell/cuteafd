use super::*;
use crate::shared::experts::layer::ExpertLayer;
use crate::shared::memory::HostAllocation;
use cuteafd_transport::{
    expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16, ExpertProtocolV2Request,
    ExpertProtocolV2ResponseView, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
    ExpertV2SourceKind, EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM,
};
use sha2::{Digest, Sha256};

fn reference(fixture: &Path, aot: &Path, capacity: usize) -> Result<Vec<Vec<u8>>> {
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture.join("fixture.json"))?)?;
    let export: serde_json::Value = serde_json::from_slice(&std::fs::read(
        aot.join(format!("m{capacity}/v41_exl3.json")),
    )?)?;
    ensure!(
        meta["canonical_routes"] == true
            && meta["input_format"] == "fp8_k32"
            && meta["output_dtype"] == "bf16"
            && meta["layer"] == "layers.0"
            && meta["slice_start"] == 1280
            && meta["width"] == 512
            && meta["capacity"] == capacity,
        "unexpected worker fixture"
    );
    for key in ["tile", "direct", "output_dtype"] {
        ensure!(meta[key] == export[key], "fixture/export policy mismatch");
    }
    let mut payloads = Vec::new();
    for name in ["input", "ids", "weights", "expected"] {
        let bytes = std::fs::read(fixture.join(format!("{name}.bin")))?;
        ensure!(
            Some(bytes.len() as u64) == meta["artifacts"][name]["bytes"].as_u64()
                && Some(format!("{:x}", Sha256::digest(&bytes)).as_str())
                    == meta["artifacts"][name]["sha256"].as_str(),
            "corrupt worker fixture"
        );
        payloads.push(bytes);
    }
    Ok(payloads)
}

#[test]
fn compact_tp3_ownership_preserves_fp8_routes_and_rejects_canonical_frames() -> Result<()> {
    let topology = SparkTopology::NATIVE_TP3_EP1;
    let mut owned = ExpertProtocolV2Request::new(91, 17, 0, 5120,
        ExpertV2Dtype::Fp8E4m3Ue8m0K32,
        vec![ExpertProtocolV2RowDescriptor { row_id: 0, source_kind: ExpertV2SourceKind::Prefill,
            source_request_id: 1, token_position: 0, route_offset: 0, route_count: 6 }],
        (0..6).map(|expert| ExpertProtocolV2RouteEntry {
            row_index: 0, expert_id: expert * 61, gate_weight: (expert + 1) as f32 / 32.0,
        }).collect(), vec![0x5a; 5280])?;
    owned.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    let canonical_frame = owned.encode()?;
    let canonical = BackboneRequest::parse(&canonical_frame, 1)?;
    let (mut ids, mut weights) = ([0; 6], [0.0; 6]);
    copy_routes(&canonical, None, &mut ids, &mut weights)?;
    let expected = (ids, weights.map(f32::to_bits));
    assert!(copy_routes(&canonical, Some(topology),
        &mut ids, &mut weights).is_err());
    owned.with_native_group_owners(&[0; 384], topology)?;
    let frame = owned.encode()?;
    let request = BackboneRequest::parse_native_group(&frame, 1, topology)?;
    assert_eq!(request.hidden(), canonical.hidden());
    for rank in 0..3 {
        copy_routes(&request, Some(topology), &mut ids, &mut weights)?;
        assert_eq!((ids, weights.map(f32::to_bits)), expected);
        assert_eq!(topology.executor_id(rank)?, 7 + rank as u64);
    }
    assert!(copy_routes(&request, None, &mut ids, &mut weights).is_err());
    assert!(copy_routes(&request, Some(SparkTopology::NATIVE_TP3_EP2),
        &mut ids, &mut weights).is_err());
    Ok(())
}

#[test]
fn exl3_worker_rank_bounds_include_tp1() -> Result<()> {
    for world in 1..=6 {
        for rank in 0..world {
            Exl3Worker::validate_rank(world, rank)?;
        }
        assert!(Exl3Worker::validate_rank(world, world).is_err());
    }
    for world in [0, 7, 8, 9] {
        assert!(Exl3Worker::validate_rank(world, 0).is_err());
    }
    Ok(())
}

fn sparse_layout(layer: cuteafd_loader::V41Exl3Layer) -> cuteafd_loader::V41Exl3Residency {
    cuteafd_loader::V41Exl3Residency {
        layer, world: 4, rank: 3,
        intermediate_start: 1536, experts: 256, intermediate: 512, tiers: vec![4],
        projection_counts: vec![[256; 3]], buffers: vec![], loads: vec![],
    }
}

#[test]
fn exl3_worker_sparse_owner_ids_preserve_absolute_request_mapping() -> Result<()> {
    use cuteafd_loader::V41Exl3Layer::Backbone;
    let layouts = [sparse_layout(Backbone(13)), sparse_layout(Backbone(15))];
    let ids = Exl3Worker::resident_layers(&layouts)?;
    assert_eq!(ids, [13, 15]);
    Exl3Worker::validate_layer(&ids, 0, 13)?;
    Exl3Worker::validate_layer(&ids, 1, 15)?;
    for (index, requested) in [(0, 14), (1, 14), (1, 13), (2, 15)] {
        assert!(Exl3Worker::validate_layer(&ids, index, requested).is_err());
    }
    Ok(())
}

#[test]
fn exl3_worker_sparse_owners_reject_invalid_ids_and_geometry() {
    use cuteafd_loader::V41Exl3Layer::{Backbone, Dspark};
    assert!(Exl3Worker::resident_layers(std::iter::empty()).is_err());
    for layers in [[Backbone(13), Backbone(13)], [Backbone(15), Backbone(13)],
        [Backbone(13), Dspark(0)]] {
        assert!(Exl3Worker::resident_layers(&layers.map(sparse_layout)).is_err());
    }
    for change in 0..6 {
        let mut layouts = [sparse_layout(Backbone(13)), sparse_layout(Backbone(15))];
        let second = &mut layouts[1];
        match change {
            0 => second.rank = 2,
            1 => second.world = 6,
            2 => second.intermediate_start = 1024,
            3 => second.intermediate = 384,
            4 => second.experts = 128,
            _ => second.tiers = vec![3, 4],
        }
        assert!(Exl3Worker::resident_layers(&layouts).is_err());
    }
}

#[test]
fn exl3_worker_host_path_switch() -> Result<()> {
    assert_eq!(HostPath::parse(None)?, HostPath::Async);
    assert_eq!(HostPath::parse(Some(""))?, HostPath::Async);
    assert_eq!(HostPath::parse(Some("async"))?, HostPath::Async);
    assert_eq!(HostPath::parse(Some("blocking"))?, HostPath::Blocking);
    assert!(HostPath::parse(Some("spin")).is_err());
    Ok(())
}

#[test]
fn exl3_route_dump_records_match_the_benchmark_format() -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("routes.3.bin");
    {
        let mut dump = RouteDump::open(&path, 2)?;
        dump.record(7, 1, 8, &[0, 1, 2, 3, 4, 5, 6, 287], &[0.5; 8])?;
        dump.record(8, 2, 8, &[9; 16], &[0.25; 16])?;
        // Past the call limit: dropped.
        dump.record(9, 1, 8, &[1; 8], &[1.0; 8])?;
    }
    let bytes = std::fs::read(&path)?;
    let words: Vec<u32> = bytes.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect();
    assert_eq!(words.len(), (4 + 16) + (4 + 32));
    assert_eq!(&words[..4], &[0x3145_5452, 7, 1, 8]);
    assert_eq!(words[4 + 7], 287);
    assert_eq!(f32::from_bits(words[4 + 8]), 0.5);
    assert_eq!(&words[20..24], &[0x3145_5452, 8, 2, 8]);
    assert_eq!(f32::from_bits(words[24 + 16]), 0.25);
    Ok(())
}

#[test]
fn exl3_worker_capacity_bounds() -> Result<()> {
    assert_eq!(Exl3Worker::capacities(1)?, vec![1]);
    assert_eq!(Exl3Worker::capacities(16)?, vec![1, 16, 80]);
    assert_eq!(Exl3Worker::capacities(40)?, vec![1, 16, 80]);
    assert_eq!(
        Exl3Worker::capacities(4096)?,
        vec![1, 16, 80, 256, 1024, 4096]
    );
    assert!(Exl3Worker::capacities(0).is_err());
    assert!(Exl3Worker::capacities(4097).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA, native EXL3 wire decoder, full snapshot, AOT and canonical wire fixture"]
fn exl3_worker_mapped_and_chunked_responses_match_reference() -> Result<()> {
    let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    lib.cuda_set_device(0)?;
    let catalog = cuteafd_loader::read_official_v41_catalog(Path::new(&std::env::var("CUTEAFD_EXL3_SNAPSHOT")?),
    )?;
    let aot = std::path::PathBuf::from(std::env::var("CUTEAFD_EXL3_AOT")?);
    let fixture = std::path::PathBuf::from(std::env::var("CUTEAFD_EXL3_FIXTURE")?);
    let layer = ExpertLayer::Backbone { layer: 0, rank: 2 };
    let budget = Exl3Weights::plan(&catalog, layer)?;
    let (free, _) = lib.cuda_memory_info()?;
    ensure!(
        free > budget.resident_bytes + Exl3Worker::plan(&aot, 4096, Exl3Schedule::Default)? + (256 << 20),
        "insufficient GPU headroom"
    );
    let weights = Rc::new(vec![Exl3Weights::load(
        &lib,
        &catalog,
        layer,
        budget.resident_bytes,
    )?]);
    let workspace = Exl3Worker::plan(&aot, 4096, Exl3Schedule::Default)?;
    let mut worker = Exl3Worker::new(&lib, weights, &aot, 4096, workspace, Exl3Schedule::Default)?;
    let mut exchange = HostExpertExchange::new(4096)?;
    let mut row_indices = vec![0; 4096];
    assert!(worker.bind_layer(1).is_err());
    worker.bind_layer(0)?;
    assert_eq!(
        worker
            .executions
            .iter()
            .map(|e| e.capacity())
            .collect::<Vec<_>>(),
        vec![1, 16, 80, 256, 1024, 4096]
    );
    for rows in [1u32, 3, 16, 17, 80, 81, 256, 257, 1024, 1025, 4096, 1] {
        let capacity = [1usize, 16, 80, 256, 1024, 4096]
            .into_iter()
            .find(|&c| c >= rows as usize)
            .unwrap();
        let payloads = reference(&fixture.join(format!("m{capacity}")), &aot, capacity)?;
        let mut owned = ExpertProtocolV2Request::new(
            91,
            17,
            0,
            5120,
            ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..rows)
                .map(|r| ExpertProtocolV2RowDescriptor {
                    row_id: r as u64,
                    source_kind: ExpertV2SourceKind::Prefill,
                    source_request_id: 1,
                    token_position: r as u64,
                    route_offset: r * 6,
                    route_count: 6,
                })
                .collect(),
            (0..rows as usize * 6)
                .map(|r| ExpertProtocolV2RouteEntry {
                    row_index: r as u32 / 6,
                    expert_id: u32::from_le_bytes(
                        payloads[1][r * 4..r * 4 + 4].try_into().unwrap(),
                    ),
                    gate_weight: f32::from_le_bytes(
                        payloads[2][r * 4..r * 4 + 4].try_into().unwrap(),
                    ),
                })
                .collect(),
            payloads[0][..rows as usize * 5280].to_vec(),
        )?;
        owned.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        let frame = owned.encode()?;
        let request = BackboneRequest::parse(&frame, 4096)?;
        let bytes = request.plane_bytes()?;
        let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let mut host = HostAllocation::new(&lib, prefix + bytes + 64)?;
        host.bytes_mut().fill(0xa5);
        let alias = lib.cuda_host_buffer_device_alias(host.buffer)?;
        let short = CuteafdDeviceBuffer {
            bytes: prefix + bytes - 1,
            ..alias
        };
        assert!(
            unsafe { worker.execute_mapped_request(&request, 3, &mut exchange, short, None)? }.is_none()
        );
        assert!(host.bytes_mut().iter().all(|&b| b == 0xa5));
        assert!(
            unsafe { worker.execute_mapped_request(&request, 2, &mut exchange, alias, None) }.is_err()
        );
        assert!(host.bytes_mut().iter().all(|&b| b == 0xa5));
        let response =
            unsafe { worker.execute_mapped_request(&request, 3, &mut exchange, alias, None)? }.unwrap();
        assert_eq!(response.header.executor_id, 3);
        assert_eq!(response.partial_output_payload.bytes, bytes);
        assert!(host.bytes_mut()[..prefix].iter().all(|&b| b == 0xa5));
        assert!(host.bytes_mut()[prefix + bytes..]
            .iter()
            .all(|&b| b == 0xa5));
        assert_eq!(
            &host.bytes_mut()[prefix..prefix + bytes],
            &payloads[3][..bytes]
        );
        // Debug-checksum requests must use host encoding; force multiple frames.
        owned.header.flags |= EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM;
        let frame = owned.encode()?;
        let request = BackboneRequest::parse(&frame, 4096)?;
        assert!(
            unsafe { worker.execute_mapped_request(&request, 3, &mut exchange, alias, None)? }.is_none()
        );
        let max_frame =
            cuteafd_transport::EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN + 3 * (10240 + 4);
        let mut actual = Vec::new();
        let mut chunks = 0;
        worker.execute_host_chunks(
            &request,
            3,
            &mut exchange,
            &mut row_indices,
            max_frame,
            |response| {
                let encoded = response.to_owned()?.encode()?;
                assert!(encoded.len() <= max_frame);
                let parsed = ExpertProtocolV2ResponseView::parse(&encoded)?;
                if let Some(indices) = response.row_indices {
                    assert_eq!(indices[0] as usize, actual.len() / 10240);
                }
                actual.extend_from_slice(parsed.partial_output_payload());
                chunks += 1;
                Ok(())
            },
        )?;
        assert_eq!(actual, payloads[3][..bytes]);
        assert_eq!(chunks, rows.div_ceil(3));
        // Sink failure occurs after GPU completion and does not poison later work.
        assert!(worker
            .execute_host_chunks(
                &request,
                3,
                &mut exchange,
                &mut row_indices,
                max_frame,
                |_| anyhow::bail!("intentional send failure")
            )
            .is_err());
        owned.header.layer_id = 1;
        let bad_frame = owned.encode()?;
        let bad = BackboneRequest::parse(&bad_frame, 4096)?;
        assert!(worker
            .execute_host_chunks(&bad, 3, &mut exchange, &mut row_indices, max_frame, |_| Ok(
                ()
            ))
            .is_err());
        println!("PASS EXL3 worker maximum=4096 selected_capacity={capacity} rows={rows}: native mapped BF16 output equals B12x, guards, chunked checksum fallback, bounds/identity/layer rejection and sink-failure recovery");
    }
    println!("EXL3 worker workspace payload: {workspace} bytes; 384 resident experts, six routed IDs, TP4 rank 2; host architecture {}", std::env::consts::ARCH);
    Ok(())
}

#[test]
fn gb10_schedule_runs_the_glm_flash_decode_exports_and_refuses_mislabels() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (gb10, default) = (Exl3Schedule::Gb10, Exl3Schedule::Default);
    for capacity in [1, 80] {
        assert_eq!(gb10.directory(root.path(), capacity), root.path().join(format!("m{capacity}-gb10")));
    }
    for capacity in [1, 16, 80, 256, 1024, 4096] {
        assert_eq!(default.directory(root.path(), capacity), root.path().join(format!("m{capacity}")));
        if ![1, 80].contains(&capacity) {
            // Prefill capacities (and the unused m16) run their default export.
            assert_eq!(gb10.directory(root.path(), capacity), root.path().join(format!("m{capacity}")));
        }
    }
    // GB10 exports exist for GLM 5.3 Flash only.
    gb10.validate(Exl3RowPolicy::GlmFlashK64)?;
    assert!(gb10.validate(Exl3RowPolicy::Nearest).is_err());
    default.validate(Exl3RowPolicy::Nearest)?;
    let write = |name: &str, schedule: Option<&str>| -> Result<()> {
        let directory = root.path().join(name);
        std::fs::create_dir_all(&directory)?;
        let mut meta = serde_json::json!({"schema": "cuteafd.v41-exl3-aot.v1", "capacity": 80});
        if let Some(schedule) = schedule { meta["decode_schedule"] = schedule.into(); }
        std::fs::write(directory.join("v41_exl3.json"), serde_json::to_vec(&meta)?)?;
        Ok(())
    };
    let canonical = "l2=2,pf1=4,pf2=8,pdl=2";
    for capacity in [1, 80] {
        write(&format!("m{capacity}"), None)?;
        write(&format!("m{capacity}-gb10"), Some(canonical))?;
    }
    for capacity in [1, 80] {
        default.check_export(root.path(), capacity)?;
        gb10.check_export(root.path(), capacity)?;
    }
    // A scheduled directory without its schedule, a default one with one, a missing one.
    write("m80-gb10", None)?;
    assert!(gb10.check_export(root.path(), 80).is_err());
    write("m1", Some(canonical))?;
    assert!(default.check_export(root.path(), 1).is_err());
    std::fs::remove_dir_all(root.path().join("m1-gb10"))?;
    assert!(gb10.check_export(root.path(), 1).is_err());
    Ok(())
}
