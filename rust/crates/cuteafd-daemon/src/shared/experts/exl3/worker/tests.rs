use super::*;

#[test]
fn exl3_worker_checks_the_selected_response_destination() -> Result<()> {
    let host_bytes = 2752 * 4096 * 2;
    // Muse returns eight per-expert BF16 rows rather than a compact reduced row.
    let partial_bytes = 512 * 8 * 4096 * 2;
    assert!(partial_bytes > host_bytes);
    let slot_bytes = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN + partial_bytes;
    let payload_bytes = slot_bytes - EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
    Exl3Worker::validate_output_capacity(partial_bytes, host_bytes, Some(payload_bytes))?;
    assert!(Exl3Worker::validate_output_capacity(partial_bytes, host_bytes, Some(payload_bytes - 1)).is_err());
    assert!(Exl3Worker::validate_output_capacity(partial_bytes, host_bytes, None).is_err());
    // A large host fallback cannot make an undersized mapped destination safe.
    assert!(Exl3Worker::validate_output_capacity(partial_bytes, partial_bytes, Some(payload_bytes - 1)).is_err());
    Exl3Worker::validate_output_capacity(host_bytes, host_bytes, None)?;
    Exl3Worker::validate_output_capacity(host_bytes, 0, Some(host_bytes))?;
    Ok(())
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

fn sparse_layout(layer: cuteafd_loader::V41Exl3Layer) -> cuteafd_loader::V41Exl3Residency {
    cuteafd_loader::V41Exl3Residency {
        layer, world: 4, rank: 3, codebook: cuteafd_loader::Exl3Codebook::Mcg,
        intermediate_start: 1536, experts: 256, intermediate: 512, tiers: vec![4],
        projection_counts: vec![[256; 3]], unit_map: None, buffers: vec![], loads: vec![],
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
        vec![1, 16, 80, 256, 1024, 2048, 2752, 4096]
    );
    assert_eq!(Exl3Worker::capacities(2752)?, vec![1, 16, 80, 256, 1024, 2048, 2752]);
    assert!(Exl3Worker::capacities(0).is_err());
    assert!(Exl3Worker::capacities(4097).is_err());
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
fn packaged_schedule_identity_matches_exact_variant_without_a_policy_allowlist() -> Result<()> {
    let package = tempfile::tempdir()?;
    let root = package.path().join("tp6-compact-rank0");
    let directory = root.join("m80");
    std::fs::create_dir_all(&directory)?;
    let manifest = package.path().join("manifest.json");
    let export = directory.join("v41_exl3.json");
    let write = |recorded: Option<&str>, actual: Option<&str>, label: Option<&str>| -> Result<()> {
        std::fs::write(&manifest, serde_json::to_vec(&serde_json::json!({
            "schema": "cuteafd.exl3-package.v1",
            "variants": [{"directory": "tp6-compact-rank0/m80",
                "schedule": label, "decode_schedule": recorded}]
        }))?)?;
        std::fs::write(&export, serde_json::to_vec(&serde_json::json!({
            "decode_schedule": actual
        }))?)?;
        Ok(())
    };
    let schedule = Exl3Schedule::Default;
    for identity in [None, Some("l2=2,pf1=0,pf2=0,pdl=1"), Some("other-package-policy")] {
        write(identity, identity, None)?;
        schedule.check_export(&root, 80)?;
    }
    for (recorded, actual) in [(None, Some("hint")), (Some("hint"), None),
        (Some("hint"), Some("different"))] {
        write(recorded, actual, None)?;
        assert!(schedule.check_export(&root, 80).is_err());
    }
    write(Some("hint"), Some("hint"), Some("gb10"))?;
    assert!(schedule.check_export(&root, 80).is_err());
    write(Some("hint"), Some("hint"), None)?;
    let mut duplicate: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest)?)?;
    let variant = duplicate["variants"][0].clone();
    duplicate["variants"].as_array_mut().unwrap().push(variant);
    std::fs::write(&manifest, serde_json::to_vec(&duplicate)?)?;
    assert!(schedule.check_export(&root, 80).is_err());
    std::fs::write(&manifest, br#"{"schema":"cuteafd.exl3-package.v1","variants":[]}"#)?;
    assert!(schedule.check_export(&root, 80).is_err());
    // Explicit standalone roots still reject a labelled default export.
    std::fs::remove_file(manifest)?;
    assert!(schedule.check_export(&root, 80).is_err());
    Ok(())
}

/// #762: a deferred mapped request (the async exchange's lane-owned slots)
/// keeps its expert timing pending and publishes it at `complete`, after the
/// owner saw that unit's fence complete, so later units can remain queued.
#[test]
fn deferred_exl3_requests_publish_their_timing_at_completion() {
    let source = include_str!("../worker.rs");
    let deferred = &source[source.find("if self.deferred && destination.is_some() {").unwrap()..];
    let early = &deferred[..deferred.find("return Ok(());").unwrap()];
    assert!(early.contains("self.pending.record((request.request_id(), request.layer(), request.rows(), executor_id), self.stream.raw)?"));
    let complete = &source[source.find("pub(crate) fn complete(&mut self)").unwrap()..];
    let complete = &complete[..complete.find("\n    }\n").unwrap()];
    assert!(complete.contains("self.pending.pop()?") && complete.contains("timing.publish("));
    assert!(source.contains("pub(crate) fn unit_ready(&self) -> Result<bool> { self.pending.ready() }"));
    let backend = include_str!("../../service/backend.rs");
    assert!(backend.contains("Self::Exl3(execution) => execution.complete(),"));
}

/// A merged wave records one worker timing span for all its members: only the
/// first member publishes it, so later members never read unrecorded events.
#[test]
fn merged_exl3_waves_publish_one_timing_span() {
    let source = include_str!("../worker.rs");
    let launch = &source[source.find("fn launch_exl3_wave").unwrap()..];
    let launch = &launch[..launch.find("\n}\n").unwrap()];
    assert!(launch.contains("timing_units.extend((0..wave.members.len()).map(|index| index == 0));"));
    let complete = &source[source.find("pub(crate) fn complete(&mut self)").unwrap()..];
    let complete = &complete[..complete.find("\n    }\n").unwrap()];
    assert!(complete.contains("self.timing_units.pop_front().unwrap_or(true)") && complete.contains(".filter(|_| timed)"));
    assert!(source.contains("if self.waves.is_some() { self.timing_units.push_back(true); }"));
}

/// Merged-wave correctness of EXL3 staging over a real installed Spark
/// layout and its row-addressed segments program (#2191 EXL3): one layer of
/// real checkpoint weights for this TP rank; mapped FP8 K32 frames staged
/// and flushed as one wave must write every member's reply byte-identical to
/// that unit run alone through the same worker. One Spark.
#[test]
#[ignore = "requires an SM121 Spark, CUTEAFD_NATIVE_LIB, CUTEAFD_EXL3_SNAPSHOT, CUTEAFD_EXL3_LAYOUT (a Spark package layout directory with m8192-segments), CUTEAFD_EXL3_WORLD and CUTEAFD_EXL3_LAYER"]
fn exl3_merged_waves_are_byte_identical_to_singleton_units() -> Result<()> {
    use crate::shared::experts::layer::ExpertLayer;
    use cuteafd_transport::{expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16, ExpertProtocolV2Request,
        ExpertProtocolV2RequestView, ExpertProtocolV2RowDescriptor, ExpertProtocolV2RouteEntry,
        ExpertV2SourceKind, ProtocolV2RequestDevicePayload};
    let catalog = cuteafd_loader::read_expert_catalog(Path::new(&std::env::var("CUTEAFD_EXL3_SNAPSHOT")?))?;
    let geometry = catalog.routed_experts().geometry()?;
    cuteafd_core::set_expert_geometry(geometry)
        .map_err(|set| anyhow::anyhow!("process geometry already {set:?}; run this test alone"))?;
    // SAFETY: this opt-in test accepts only operator-selected trusted build artifacts.
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    library.cuda_set_device(0)?;
    let directory = std::path::PathBuf::from(std::env::var("CUTEAFD_EXL3_LAYOUT")?);
    let world: usize = std::env::var("CUTEAFD_EXL3_WORLD")?.parse()?;
    let layer: usize = std::env::var("CUTEAFD_EXL3_LAYER")?.parse()?;
    ensure!(Exl3Worker::segments_directory(&directory, Exl3Epilogue::Sum).is_some(),
        "layout {} ships no m8192-segments program", directory.display());
    let weights = Rc::new(vec![Exl3Weights::load(&library, &catalog,
        ExpertLayer::BackboneExl3Tp { layer, rank: 0, world }, usize::MAX)?]);
    // Expert timing on (as the expert-timing card profile runs it): a merged
    // wave records one span, published once, never from unrecorded events.
    let timing_dir = std::env::temp_dir().join(format!("exl3-wave-timing-{}", std::process::id()));
    std::fs::create_dir_all(&timing_dir)?;
    std::env::set_var("CUTEAFD_EXPERT_TIMING", "1");
    std::env::set_var("CUTEAFD_PROFILE_DIR", &timing_dir);
    std::env::set_var("CUTEAFD_PROFILE_HOST", "oracle");
    let unit = 2752usize;
    let extent = cuteafd_core::OwnerExtent { owner: cuteafd_core::ExecutionOwner::Prefill,
        capacity: unit as u32, compiled: 4096, staging_rows: unit as u32, buffered_units: 3 };
    let mut worker = Exl3Worker::new(&library, weights, &directory, unit as u32, usize::MAX,
        Exl3Schedule::default(), Exl3Epilogue::Sum)?;
    worker.defer(3, unit, cuteafd_core::ExecutionOwner::Prefill)?;
    worker.prepare_waves(extent)?;
    ensure!(worker.waves.is_some() && worker.segments.is_some(), "prefill owner did not prepare merged waves");
    worker.bind_layer(0)?;
    let executor = worker.executor_id;
    let (hidden, topk, experts) = (geometry.hidden as usize, geometry.topk as usize, geometry.experts as usize);
    let pitch = hidden + hidden / 32;
    let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
    let rows = [2752u32, 2752, 2487, 1900, 333];
    let mut frames = Vec::new();
    let mut replies = Vec::new();
    for (index, &count) in rows.iter().enumerate() {
        // Slots 0/1 on two hot experts (multi-block groups shared by every
        // member); the rest distinct experts per row.
        let ids = (0..count as usize * topk).map(|route| {
            let (row, slot) = (route / topk, route % topk);
            if slot < 2 { slot as u32 } else {
                (2 + (row * 13 + index * 29 + (slot - 2) * ((experts - 2) / topk).max(1)) % (experts - 2)) as u32 }
        }).collect::<Vec<_>>();
        // FP8 E4M3 values with in-range UE8M0 scales (2^-8..2^-1).
        let mut wire = vec![0u8; count as usize * pitch];
        for (row, bytes) in wire.chunks_exact_mut(pitch).enumerate() {
            for (column, value) in bytes[..hidden].iter_mut().enumerate() {
                let code = ((row * 31 + column * 7 + index * 977) % 110) as u8 + 8;
                *value = if (row + column) % 3 == 0 { code | 0x80 } else { code };
            }
            for (group, scale) in bytes[hidden..].iter_mut().enumerate() { *scale = 119 + ((row + group + index) % 8) as u8; }
        }
        let mut owned = ExpertProtocolV2Request::new(700 + index as u64, 17, layer as u32, geometry.hidden,
            ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..count).map(|row| ExpertProtocolV2RowDescriptor { row_id: u64::from(row),
                source_kind: ExpertV2SourceKind::Prefill, source_request_id: index as u64,
                token_position: u64::from(row), route_offset: row * topk as u32, route_count: topk as u32 }).collect(),
            ids.iter().enumerate().map(|(route, &expert_id)| ExpertProtocolV2RouteEntry {
                row_index: (route / topk) as u32, expert_id,
                gate_weight: ((route % topk + 1 + index) as f32) / 32.0 }).collect(), wire)?;
        owned.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        let encoded = owned.encode()?;
        let mut frame = HostAllocation::new(&library, encoded.len(), crate::shared::memory::TEST)?;
        frame.bytes_mut().copy_from_slice(&encoded);
        frames.push(frame);
        replies.push(HostAllocation::new(&library, prefix + count as usize * hidden * 2 + 64, crate::shared::memory::TEST)?);
    }
    let mut exchange = HostExpertExchange::new(unit as u32)?;
    let mut run = |members: &[usize], worker: &mut Exl3Worker<'_>| -> Result<Vec<Vec<u8>>> {
        for &member in members {
            replies[member].bytes_mut().fill(0xa5);
            let view = ExpertProtocolV2RequestView::parse(frames[member].bytes())?;
            let request = BackboneRequest::parse(frames[member].bytes(), unit as u32)?;
            let alias = library.cuda_host_buffer_device_alias(frames[member].buffer)?;
            let mapped = ProtocolV2RequestDevicePayload {
                hidden_payload: CuteafdDeviceBuffer { ptr: view.hidden_payload().as_ptr().cast_mut().cast(),
                    bytes: view.hidden_payload().len(), ..alias }, execution_lane: 0,
                response_slot: Some(library.cuda_host_buffer_device_alias(replies[member].buffer)?) };
            // SAFETY: every frame and reply outlives the drain below.
            unsafe { worker.execute_mapped_payload(&request, executor, &mut exchange, mapped)?; }
        }
        worker.flush_staged()?;
        worker.synchronize()?;
        while !worker.pending.is_empty() { ensure!(worker.unit_ready()?, "drained unit not ready"); worker.complete()?; }
        members.iter().map(|&member| {
            let end = prefix + rows[member] as usize * hidden * 2;
            let bytes = replies[member].bytes();
            ensure!(bytes[..prefix].iter().all(|&b| b == 0xa5) && bytes[end..].iter().all(|&b| b == 0xa5),
                "reply guard overwritten");
            Ok(bytes[prefix..end].to_vec())
        }).collect()
    };
    let singles = (0..rows.len()).map(|member| run(&[member], &mut worker).map(|mut out| out.remove(0)))
        .collect::<Result<Vec<_>>>()?;
    for single in &singles { ensure!(single.chunks_exact(2).any(|w| w != [0, 0]), "zero singleton reply"); }
    let merged_before = worker.waves.as_ref().unwrap().merged;
    for pair in [[0usize, 1], [1, 2], [2, 3], [3, 0], [0, 4], [4, 2]] {
        let merged = run(&pair, &mut worker)?;
        for (position, &member) in pair.iter().enumerate() {
            let different = merged[position].iter().zip(&singles[member]).filter(|(a, b)| a != b).count();
            ensure!(different == 0, "members {pair:?}: member {member} differs in {different} bytes from its singleton");
        }
        eprintln!("EXL3 merged wave {:?} ({} rows): byte-identical to singletons", pair,
            pair.iter().map(|&m| rows[m]).sum::<u32>());
    }
    ensure!(worker.waves.as_ref().unwrap().merged - merged_before == 6, "every pair must flush as one merged launch");
    drop(worker);
    let samples = std::fs::read_to_string(timing_dir.join("expert-timing-oracle-rank0.jsonl"))?;
    let samples = samples.lines().map(serde_json::from_str::<serde_json::Value>).collect::<Result<Vec<_>, _>>()?;
    let published = samples.iter().map(|sample| sample["request_id"].as_u64().unwrap_or(u64::MAX)).collect::<Vec<_>>();
    ensure!(published == [700, 701, 702, 703, 704, 700, 701, 702, 703, 700, 704],
        "expected one timing span per singleton and per merged wave, got {published:?}");
    ensure!(samples.iter().all(|sample| sample["compute_us"].as_f64().is_some_and(|us| us > 0.0)),
        "timing span without measured compute");
    std::fs::remove_dir_all(&timing_dir)?;
    eprintln!("EXL3 merged waves with expert timing: {} spans for 5 singletons + 6 waves", samples.len());
    Ok(())
}
