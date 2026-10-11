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
fn exl3_worker_rank_bounds_include_tp1() -> Result<()> {
    for world in 1..=8 {
        for rank in 0..world {
            Exl3Worker::validate_rank(world, rank)?;
        }
        assert!(Exl3Worker::validate_rank(world, world).is_err());
    }
    for world in [0, 9] {
        assert!(Exl3Worker::validate_rank(world, 0).is_err());
    }
    Ok(())
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
    assert_eq!(Exl3Worker::capacities(16)?, vec![1, 16]);
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
    let catalog = cuteafd_loader::read_official_v41_catalog(
        cuteafd_loader::OFFICIAL_V41_MODEL_ID,
        Path::new(&std::env::var("CUTEAFD_EXL3_SNAPSHOT")?),
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
fn paired_worker_loading_rejects_wrong_rank_and_mixed_capacities() -> Result<()> {
    use cuteafd_loader::V41Exl3Partition;
    let root = tempfile::tempdir()?;
    let original = serde_json::json!({
        "schema":"cuteafd.v41-exl3-aot.v1", "output_dtype":"bf16", "sparkinfer_revision":"test",
        "hidden":5120,"intermediate":640,"experts":384,"capacity":80,"top_k":6,
        "bits":[3,4],"swiglu_limit":10.0,"direct":false,"sms":48,"blocks_per_sm":1,
        "buffers":{},"objects":[],"trellis_lut":{"file":"lut","bytes":16,"sha256":"test"}
    });
    let write = |capacity, value: &serde_json::Value| -> Result<()> {
        let directory = root.path().join(format!("m{capacity}"));
        std::fs::create_dir_all(&directory)?;
        std::fs::write(directory.join("v41_exl3.json"), serde_json::to_vec(value)?)?;
        Ok(())
    };
    for c in [1, 16, 80] { write(c, &original)?; }
    for rank in 0..4 {
        assert_eq!(Exl3Worker::partition(root.path(), 80, rank, Exl3Schedule::Default)?, V41Exl3Partition::Disjoint);
    }
    for (boundary, ranks) in [("last", [0, 2]), ("first", [1, 3])] {
        let mut paired = original.clone();
        paired["paired_boundary"] = boundary.into();
        paired["descriptor_rows"] = 4.into();
        paired["native_info_version"] = 3.into();
        for c in [1, 16, 80] { write(c, &paired)?; }
        for rank in ranks {
            assert_eq!(Exl3Worker::partition(root.path(), 80, rank, Exl3Schedule::Default)?, V41Exl3Partition::PairedTp4);
            assert!(Exl3Worker::partition(root.path(), 80, rank ^ 1, Exl3Schedule::Default).is_err());
        }
        write(80, &original)?;
        assert!(Exl3Worker::partition(root.path(), 80, ranks[0], Exl3Schedule::Default).is_err());
        assert_eq!(Exl3Worker::partition(root.path(), 16, ranks[0], Exl3Schedule::Default)?, V41Exl3Partition::PairedTp4);
    }
    assert!(Exl3Worker::partition(root.path(), 80, 4, Exl3Schedule::Default).is_err());
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

#[test]
#[ignore = "requires paired Spark package, paired reference fixtures, snapshot and CUDA"]
fn paired_worker_mapped_and_chunked_match_reference() -> Result<()> {
    use cuteafd_loader::V41Exl3Partition;
    use cuteafd_transport::expert::{V41PairedRouteWord, V41_EXL3_PAIRED_REQUEST_FLAG};
    let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    lib.cuda_set_device(0)?;
    let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_EXL3_SNAPSHOT")?);
    let catalog = cuteafd_loader::read_official_v41_catalog(cuteafd_loader::OFFICIAL_V41_MODEL_ID, &snapshot)?;
    let package = std::path::PathBuf::from(std::env::var("CUTEAFD_EXL3_PAIRED_PACKAGE")?);
    let fixtures = std::path::PathBuf::from(std::env::var("CUTEAFD_EXL3_PAIRED_FIXTURES")?);
    for (rank, boundary) in [(0, "last"), (1, "first")] {
        let fixture = fixtures.join(format!("paired-fixture-{boundary}"));
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture.join("fixture.json"))?)?;
        ensure!(meta["layer"] == "layers.30" && meta["paired_rank"] == rank
            && meta["paired_boundary"] == boundary && meta["capacity"] == 80
            && meta["input_format"] == "fp8_k32" && meta["canonical_routes"] == true
            && meta["snapshot_revision"].as_str() == snapshot.file_name().and_then(|v| v.to_str()), "paired fixture identity mismatch");
        let read = |name: &str| -> Result<Vec<u8>> {
            let bytes = std::fs::read(fixture.join(format!("{name}.bin")))?;
            ensure!(Some(bytes.len() as u64) == meta["artifacts"][name]["bytes"].as_u64()
                && Some(format!("{:x}", Sha256::digest(&bytes)).as_str()) == meta["artifacts"][name]["sha256"].as_str(), "paired fixture checksum mismatch");
            Ok(bytes)
        };
        let input = read("input")?; let ids = read("ids")?; let routing = read("weights")?;
        let owners = read("owners")?; let expected = read("expected")?;
        let aot = package.join(format!("tp4-rank{rank}"));
        assert_eq!(Exl3Worker::partition(&aot, 80, rank, Exl3Schedule::Default)?, V41Exl3Partition::PairedTp4);
        let layer = ExpertLayer::Backbone { layer:30, rank };
        let budget = Exl3Weights::plan_with_layout(&catalog, layer, V41Exl3Partition::PairedTp4)?;
        let weights = Rc::new(vec![Exl3Weights::load_with_layout(&lib, &catalog, layer,
            budget.resident_bytes, V41Exl3Partition::PairedTp4)?]);
        let mut worker = Exl3Worker::new(&lib, weights, &aot, 80, Exl3Worker::plan(&aot, 80, Exl3Schedule::Default)?,
            Exl3Schedule::Default)?;
        let mut exchange = HostExpertExchange::new(80)?;
        let mut request = ExpertProtocolV2Request::new(91,17,30,5120,ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..80).map(|r| ExpertProtocolV2RowDescriptor { row_id:r as u64, source_kind:ExpertV2SourceKind::Prefill,
                source_request_id:1,token_position:r as u64,route_offset:r*6,route_count:6 }).collect(),
            (0..480).map(|r| {
                let id = u32::from_le_bytes(ids[r*4..r*4+4].try_into().unwrap());
                let owner = u32::from_le_bytes(owners[id as usize*4..id as usize*4+4].try_into().unwrap());
                Ok(ExpertProtocolV2RouteEntry { row_index:r as u32/6,
                    expert_id:V41PairedRouteWord { expert_id:id, owners:u8::try_from(owner)? }.encode()?,
                    gate_weight:f32::from_le_bytes(routing[r*4..r*4+4].try_into().unwrap()) })
            }).collect::<Result<Vec<_>>>()?, input)?;
        request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16 | V41_EXL3_PAIRED_REQUEST_FLAG;
        let frame = request.encode()?;
        let parsed = BackboneRequest::parse_paired(&frame,80)?;
        let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let mut host = HostAllocation::new(&lib,prefix+expected.len()+64)?;
        host.bytes_mut().fill(0xa5);
        let alias = lib.cuda_host_buffer_device_alias(host.buffer)?;
        let response = unsafe { worker.execute_mapped_request(&parsed,rank as u64+1,&mut exchange,alias,None)? }.context("paired mapped response absent")?;
        assert_eq!(response.header.flags & V41_EXL3_PAIRED_REQUEST_FLAG,0);
        assert_eq!(&host.bytes_mut()[prefix..prefix+expected.len()], expected.as_slice());
        assert!(host.bytes_mut()[..prefix].iter().all(|&v| v==0xa5));
        assert!(host.bytes_mut()[prefix+expected.len()..].iter().all(|&v| v==0xa5));
        request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM;
        let frame = request.encode()?;
        let parsed = BackboneRequest::parse_paired(&frame,80)?;
        let mut indices = [0;3]; let mut actual = Vec::new();
        worker.execute_host_chunks(&parsed,rank as u64+1,&mut exchange,&mut indices,
            cuteafd_transport::EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN+3*(10240+4), |response| {
                response.validate()?;
                assert_eq!(response.header.flags & V41_EXL3_PAIRED_REQUEST_FLAG,0);
                actual.extend_from_slice(response.partial_output_payload); Ok(())
            })?;
        assert_eq!(actual,expected);
        // Alternate owner bits, then restore while crossing m1/m16/m80.
        // Different capacity tiles can round BF16 differently: retain the
        // predeclared real-checkpoint tolerance for those prefix comparisons.
        for (rows, flip) in [(80usize, true), (1,false), (16,false), (17,false), (80,false)] {
            let mut next = ExpertProtocolV2Request::new(92,17,30,5120,ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                request.rows[..rows].to_vec(), request.routes[..rows*6].to_vec(),
                request.hidden_payload[..rows*5280].to_vec())?;
            next.header.flags = request.header.flags;
            if flip { for route in &mut next.routes { route.expert_id ^= 3 << 9; } }
            let frame = next.encode()?;
            let parsed = BackboneRequest::parse_paired(&frame,80)?;
            actual.clear();
            worker.execute_host_chunks(&parsed,rank as u64+1,&mut exchange,&mut indices,
                cuteafd_transport::EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN+3*(10240+4), |response| {
                    actual.extend_from_slice(response.partial_output_payload); Ok(())
                })?;
            if flip {
                assert_ne!(actual,expected, "changing ownership must change the partial");
                assert!(actual.chunks_exact(2).all(|b| f32::from_bits((u16::from_le_bytes([b[0],b[1]]) as u32)<<16).is_finite()));
            } else if rows == 80 {
                assert_eq!(actual,expected, "restored ownership must reproduce the original output");
            } else {
                let mut max_error = 0f64; let mut max_reference = 0f64;
                let mut error_sq = 0f64; let mut reference_sq = 0f64;
                for (a,b) in actual.chunks_exact(2).zip(expected[..rows*10240].chunks_exact(2)) {
                    let a = f32::from_bits((u16::from_le_bytes([a[0],a[1]]) as u32)<<16) as f64;
                    let b = f32::from_bits((u16::from_le_bytes([b[0],b[1]]) as u32)<<16) as f64;
                    assert!(a.is_finite() && b.is_finite());
                    max_error = max_error.max((a-b).abs()); max_reference = max_reference.max(b.abs());
                    error_sq += (a-b).powi(2); reference_sq += b*b;
                }
                assert!(max_reference > 0. && reference_sq > 0.);
                let relative_max = max_error/max_reference;
                let relative_l2 = (error_sq/reference_sq).sqrt();
                eprintln!("paired rank {rank} rows {rows}: relative_max={relative_max} relative_l2={relative_l2}");
                assert!(relative_max <= 0.006 && relative_l2 <= 0.003);
            }
        }
        eprintln!("paired rank {rank} ({boundary}): mapped and chunked outputs match reference exactly");
    }
    Ok(())
}
