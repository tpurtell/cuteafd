use super::*;

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
        layer, world: 4, rank: 3,
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
