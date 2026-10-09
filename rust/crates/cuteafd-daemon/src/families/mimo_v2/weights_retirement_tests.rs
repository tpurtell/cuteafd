//! CPU-only native fault injection through the actual MiMo loader entry points.
//! These checks prove Rust ownership and module unload behavior, not CUDA completion.
use super::*;
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::SafetensorsTensorMetadata;

fn checkpoint(fixture: &Fixture, tensors: &[(&str, DType, &[usize])]) -> Result<Checkpoint> {
    let mut payload = Vec::new();
    let mut headers = Vec::new();
    for (name, dtype, shape) in tensors {
        let width = match dtype { DType::Bf16 => 2, DType::F32 => 4, DType::F8E4M3 => 1, _ => unreachable!() };
        let bytes = shape.iter().product::<usize>() * width;
        headers.push(CheckpointTensor { shard: "weights.bin".into(), meta: SafetensorsTensorMetadata {
            name: (*name).into(), dtype: dtype.clone(), shape: shape.to_vec(),
            byte_offset: payload.len() as u64, byte_length: bytes as u64,
        } });
        payload.resize(payload.len() + bytes, 0);
    }
    std::fs::write(fixture.directory().join("weights.bin"), payload)?;
    headers.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
    Ok(Checkpoint { snapshot: fixture.directory().into(), config: serde_json::Value::Null,
        quantize_config: None, weight_map: Default::default(), tensors: headers, missing_shards: vec![], shard_bytes: 0 })
}

fn loader<'a>(library: &'a NativeLibrary, checkpoint: &'a Checkpoint,
    formats: &'a BTreeMap<String, MimoProjectionRepresentation>) -> MimoLoader<'a> {
    MimoLoader { library, checkpoint, stream: std::ptr::null_mut(), checkpoint_tp: 1,
        fp8_head: false, output_formats: formats, fp8_scales: crate::shared::fp8_linear::Fp8Scales::Amax,
        device: 0, peers: vec![RankDevice { device: 1, stream: std::ptr::null_mut() }] }
}

fn count(events: &str, tag: char) -> usize { events.chars().filter(|&c| c == tag).count() }

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn failed_projection_drain_retains_source_destinations_and_module() -> Result<()> {
    for launch in [0, 1] {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.configure_pack(&library, launch, 1)?;
        let checkpoint = checkpoint(&fixture, &[("weight", DType::Bf16, &[8, 128])])?;
        let formats = BTreeMap::new();
        let result = loader(&library, &checkpoint, &formats).fp8_projection("weight", 1, false);
        let error = result.err().expect("unproved packing completion");
        if launch != 0 {
            assert!(format!("{error:#}").contains("FP8 quantization (rule 0) failed"), "{error:#}");
        }
        drop(library);
        let events = fixture.events()?;
        assert_eq!(count(&events, 'D'), 3, "{events}");
        assert_eq!(count(&events, 'Q'), 1, "must reach actual loader launch: {events}");
        assert!(!events.contains(['d', 'F', 'U']), "unproved completion released an owner: {events}");
        assert!(fixture.resident()?);
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn dequant_launch_failure_retains_all_owners_until_drain_is_proved() -> Result<()> {
    for drain in [0, 1] {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.configure_pack(&library, 1, drain)?;
        let checkpoint = checkpoint(&fixture, &[("a", DType::F8E4M3, &[8, 128]),
            ("a_scale_inv", DType::F32, &[1, 1])])?;
        let formats = BTreeMap::new();
        let result = loader(&library, &checkpoint, &formats).rows(&["a".into()]);
        assert!(result.is_err());
        drop(result);
        drop(library);
        let events = fixture.events()?;
        assert_eq!(count(&events, 'B'), 1, "{events}");
        assert_eq!(count(&events, 'S'), 1, "launch error must still drain: {events}");
        if drain == 0 {
            assert_eq!(count(&events, 'd'), 3, "{events}");
            assert!(events.find('S') < events.find('d'), "freed before drain: {events}");
            assert!(events.ends_with("FU"), "{events}");
            assert!(!fixture.resident()?);
        } else {
            assert!(!events.contains(['d', 'F', 'U']), "{events}");
            assert!(fixture.resident()?);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn later_tensor_error_cannot_release_an_earlier_queued_dequant() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    fixture.configure_pack(&library, 0, 1)?;
    let checkpoint = checkpoint(&fixture, &[("a", DType::F8E4M3, &[8, 128]),
        ("a_scale_inv", DType::F32, &[1, 1]), ("z", DType::F32, &[1, 128])])?;
    let formats = BTreeMap::new();
    let result = loader(&library, &checkpoint, &formats).rows(&["a".into(), "z".into()]);
    let error = result.err().expect("unsupported later tensor");
    assert!(format!("{error:#}").contains("unsupported coordinator dtype"));
    drop(library);
    let events = fixture.events()?;
    assert_eq!(count(&events, 'B'), 1, "{events}");
    assert_eq!(count(&events, 'S'), 1, "{events}");
    assert!(!events.contains(['d', 'F', 'U']), "{events}");
    assert!(fixture.resident()?);
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn split_copy_failure_drains_before_releasing_source_or_destination() -> Result<()> {
    for drain in [0, 1] {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.configure_pack(&library, 1, drain)?;
        let checkpoint = checkpoint(&fixture, &[("weight", DType::Bf16, &[8, 256])])?;
        let formats = BTreeMap::new();
        let result = loader(&library, &checkpoint, &formats).o_proj("weight", 2);
        assert!(result.is_err());
        drop(result);
        drop(library);
        let events = fixture.events()?;
        assert_eq!(count(&events, 'C'), 1, "{events}");
        assert_eq!(count(&events, 'S'), 1, "{events}");
        if drain == 0 {
            assert_eq!(count(&events, 'd'), 2, "{events}");
            assert!(events.find('S') < events.find('d'), "{events}");
            assert!(events.ends_with("FU"), "{events}");
            assert!(!fixture.resident()?);
        } else {
            assert!(!events.contains(['d', 'F', 'U']), "{events}");
            assert!(fixture.resident()?);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn failed_peer_copy_drain_retains_all_split_outputs_without_calling_free() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    fixture.configure_pack(&library, 0, 1)?;
    fixture.configure_drain_after(&library, 1)?;
    let checkpoint = checkpoint(&fixture, &[("weight", DType::Bf16, &[8, 256])])?;
    let formats = BTreeMap::new();
    // A preceding layer's unrelated owners also unwind after a loader error.
    // Their frees must not synchronize with the failed peer's queued work.
    let preceding_layer = DeviceAllocation::new(&library, 512)?;
    let preceding_pin = crate::shared::memory::HostAllocation::new(&library, 512)?;
    let result = loader(&library, &checkpoint, &formats).o_proj("weight", 2);
    assert!(result.is_err());
    drop(result);
    drop(preceding_layer);
    drop(preceding_pin);
    drop(library);
    let events = fixture.events()?;
    assert_eq!((count(&events, 'D'), count(&events, 'C'), count(&events, 'S')), (4, 2, 2), "{events}");
    assert_eq!(count(&events, 'd'), 0,
        "even a completed rank's cudaFree can synchronize with pending peer work: {events}");
    assert!(!events.contains(['F', 'U']), "{events}");
    assert!(fixture.resident()?);
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn successful_loaders_reclaim_storage_and_unload_normally() -> Result<()> {
    for path in 0..3 {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let checkpoint = if path == 0 {
            checkpoint(&fixture, &[("weight", DType::F8E4M3, &[8, 128]),
                ("weight_scale_inv", DType::F32, &[1, 1])])?
        } else { checkpoint(&fixture, &[("weight", DType::Bf16, &[8, 256])])? };
        let formats = BTreeMap::new();
        let loader = loader(&library, &checkpoint, &formats);
        let preceding_layer = DeviceAllocation::new(&library, 512)?;
        let preceding_pin = crate::shared::memory::HostAllocation::new(&library, 512)?;
        match path {
            0 => drop(loader.rows(&["weight".into()])?),
            1 => drop(loader.o_proj("weight", 2)?),
            _ => drop(loader.fp8_projection("weight", 1, false)?),
        }
        drop(preceding_layer);
        drop(preceding_pin);
        drop(library);
        let events = fixture.events()?;
        assert_eq!(count(&events, 'D'), count(&events, 'd'), "{events}");
        assert!(events.find('S') < events.find('d'), "{events}");
        assert!(events.ends_with("FU"), "{events}");
        assert!(!fixture.resident()?);
    }
    Ok(())
}
