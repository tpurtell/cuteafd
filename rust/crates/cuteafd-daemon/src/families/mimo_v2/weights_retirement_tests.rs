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
    Ok(Checkpoint { view: None, snapshot: fixture.directory().into(), config: serde_json::Value::Null,
        quantize_config: None, weight_map: Default::default(), tensors: headers, missing_shards: vec![], shard_bytes: 0 })
}

fn loader<'a>(library: &'a NativeLibrary, checkpoint: &'a Checkpoint,
    formats: &'a BTreeMap<String, MimoProjectionRepresentation>) -> MimoLoader<'a> {
    MimoLoader { library, checkpoint, stream: std::ptr::null_mut(), checkpoint_tp: 1,
        fp8_head: false, output_formats: formats, fp8_scales: crate::shared::fp8_linear::Fp8Scales::Amax,
        device: 0, peers: vec![RankDevice { device: 1, stream: std::ptr::null_mut() }],
        dense: std::rc::Rc::new(crate::shared::experts::rtx::dense::DenseLoad::mount(library)), transient: Default::default() }
}

fn count(events: &str, tag: char) -> usize { events.chars().filter(|&c| c == tag).count() }

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn predictor_and_embedding_record_before_commit_and_replay_mount_bytes() -> Result<()> {
    use crate::shared::{experts::rtx::dense, memory::device::Device,
        token_io::{EmbedPlacement, EmbedSource, TokenEmbedding}};
    use std::os::unix::fs::FileExt;
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    fixture.immediate_copies(&library)?;
    for device in 0..2 { fixture.configure_memory(&library, device, 1 << 30)?; }
    let cfg = MimoV2Config::from_hf(&serde_json::json!({
        "model_type": "mimo_v2", "vocab_size": 256, "hidden_size": 256,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
        "swa_num_key_value_heads": 1, "head_dim": 128, "v_head_dim": 128,
        "rope_theta": 10000.0, "swa_rope_theta": 10000.0, "sliding_window": 128,
        "intermediate_size": 256, "n_routed_experts": 2, "num_experts_per_tok": 1,
        "moe_intermediate_size": 256
    }))?;
    let p = "model.mtp.layers.0";
    let mut tensors = vec![
        ("model.embed_tokens.weight".to_string(), DType::Bf16, vec![256, 256]),
        ("lm_head.weight".to_string(), DType::Bf16, vec![256, 256]),
        ("model.norm.weight".to_string(), DType::Bf16, vec![256]),
    ];
    for (name, shape) in [
        ("eh_proj.weight", vec![256, 512]), ("enorm.weight", vec![256]),
        ("hnorm.weight", vec![256]), ("final_layernorm.weight", vec![256]),
        ("input_layernorm.weight", vec![256]), ("pre_mlp_layernorm.weight", vec![256]),
        ("self_attn.o_proj.weight", vec![256, 256]), ("self_attn.attention_sink_bias", vec![2]),
    ] { tensors.push((format!("{p}.{name}"), DType::Bf16, shape)); }
    for (name, rows) in [("self_attn.qkv_proj.weight", 512), ("mlp.gate_proj.weight", 256),
        ("mlp.up_proj.weight", 256), ("mlp.down_proj.weight", 256)] {
        tensors.push((format!("{p}.{name}"), DType::F8E4M3, vec![rows, 256]));
        tensors.push((format!("{p}.{name}_scale_inv"), DType::F32, vec![rows / 128, 2]));
    }
    let refs = tensors.iter().map(|(n, d, s)| (n.as_str(), d.clone(), s.as_slice())).collect::<Vec<_>>();
    let checkpoint = checkpoint(&fixture, &refs)?;
    let path = checkpoint.snapshot.join("weights.bin");
    let file = std::fs::File::options().write(true).open(&path)?;
    for tensor in &checkpoint.tensors {
        let bytes = match tensor.meta.dtype {
            DType::Bf16 => (0..tensor.meta.byte_length / 2).flat_map(|i|
                (((0.125 * (1 + i % 7) as f32).to_bits() >> 16) as u16).to_le_bytes()).collect::<Vec<_>>(),
            DType::F32 => (0..tensor.meta.byte_length / 4).flat_map(|i|
                (0.25 * (1 + i % 5) as f32).to_le_bytes()).collect(),
            DType::F8E4M3 => (0..tensor.meta.byte_length).map(|i| 0x20 + (i % 24) as u8).collect(),
            _ => unreachable!(),
        };
        file.write_all_at(&bytes, tensor.meta.byte_offset)?;
    }
    let payload = std::fs::read(&path)?;
    let formats = BTreeMap::from([(format!("{p}.self_attn.o_proj.weight"), MimoProjectionRepresentation::Fp8)]);
    let mounted_loader = loader(&library, &checkpoint, &formats);
    let mounted = mounted_loader.mtp_weights(&cfg, 1)?;
    let digests = |weights: &MtpWeights<'_>| -> Result<BTreeMap<String, String>> {
        let mut out = BTreeMap::new();
        for (name, mut buffer) in weights.layer.buffers() {
            if name == "sinks" { buffer.bytes = 4; }
            out.insert(name.to_string(), super::super::coordinator_golden_tests::digest(&library, buffer)?);
        }
        for (i, extra) in weights.extras.iter().enumerate() {
            out.insert(format!("extra{i}"), super::super::coordinator_golden_tests::digest(&library, extra.buffer)?);
        }
        Ok(out)
    };
    let expected = digests(&mounted[0])?;
    let (mounted_target, _) = mounted_loader.model(&cfg, 0)?;
    let MimoHead::Bf16(mounted_head) = &mounted_target.head else { unreachable!() };
    let expected_head = super::super::coordinator_golden_tests::digest(&library, mounted_head.buffer)?;
    let tensor = checkpoint.tensors.iter().find(|t| t.meta.name == "model.embed_tokens.weight").unwrap();
    let source = EmbedSource::new(&checkpoint.snapshot, &tensor.shard, &tensor.meta, 256)?;
    let (mounted_embedding, ()) = TokenEmbedding::load(&library, source.clone(), EmbedPlacement::Gpu, || Ok(()))?;
    assert_eq!(mounted_embedding.verify_resident()?, 256 * 256 * 2);
    // Recording must succeed with missing payload bytes, on either target layout.
    std::fs::write(&path, [])?;
    for (peers, fp8_head) in [(0, false), (1, false), (0, true), (1, true)] {
        let load = std::rc::Rc::new(dense::DenseLoad::push(&library));
        let mut planning = loader(&library, &checkpoint, &formats);
        planning.peers.truncate(peers);
        planning.fp8_head = fp8_head;
        planning.dense = load.clone();
        let empty = planning.mtp_weights(&cfg, 0)?;
        assert!(empty.is_empty());
        assert_eq!(load.sink.as_ref().unwrap().take_all().summary(), (0, 0, 0, 0, 0, 0));
        let error = planning.mtp_weights(&cfg, 2).err().expect("unavailable predictor must refuse before loading");
        assert!(format!("{error:#}").contains("checkpoint has 1 MTP layers"));
        assert_eq!(load.sink.as_ref().unwrap().take_all().summary(), (0, 0, 0, 0, 0, 0));
        let ((embedding, (target, predictors)), mut records) = dense::recording(||
            planning.with_embedding(&cfg, 0, 1, source.clone(), EmbedPlacement::Gpu))?;
        records.extend(load.sink.as_ref().unwrap().take_all());
        assert!(target.1.is_empty() || target.1.iter().all(Vec::is_empty));
        assert_eq!(predictors.len(), 1);
        assert!(!predictors[0].layer.split);
        assert!(predictors[0].layer.buffers().iter().all(|(_, b)| b.device_id == 0));
        assert!(predictors[0].extras.iter().all(|w| w.buffer.device_id == 0));
        assert!(records.operands.iter().all(|o| o.device == 0));
        assert!(records.steps.iter().all(|s| s.device == 0 && s.bytes() <= dense::STEP_BYTES));
        let names = records.operands.iter().flat_map(|o| o.lands.iter().map(|(rect, _)| rect.name.clone()))
            .chain(records.steps.iter().flat_map(|s| s.rects.iter().map(|rect| rect.name.clone())))
            .collect::<std::collections::BTreeSet<_>>();
        assert!(names.contains("model.embed_tokens.weight"));
        for (name, _, _) in &tensors {
            assert!(names.contains(name), "unrecorded operand {name}");
        }
        assert!(!records.finishes.is_empty());
        assert!(!planning.transient.borrow().is_empty(), "scale grids must survive until replay drains");
        std::fs::write(&path, &payload)?;
        let sockets = tempfile::tempdir_in(fixture.directory())?;
        let _committed = crate::shared::experts::rtx::push::tests::receive_pushed(
            Device { library: &library, id: 0 }, None, &checkpoint.snapshot, sockets.path(), &records, None, 0, 0)?;
        planning.release_transient();
        assert_eq!(digests(&predictors[0])?, expected);
        if !fp8_head {
            let MimoHead::Bf16(head) = &target.0.head else { panic!("Flash must retain its BF16 head") };
            assert_eq!(super::super::coordinator_golden_tests::digest(&library, head.buffer)?, expected_head);
            assert!(records.operands.iter().any(|o| o.buffer.ptr == head.buffer.ptr
                && o.lands.iter().any(|(rect, _)| rect.name == "lm_head.weight")),
                "BF16 head must land directly through the committed push group");
        }
        assert_eq!(embedding.verify_resident()?, 256 * 256 * 2);
        std::fs::write(&path, [])?;
    }
    Ok(())
}

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
fn bf16_output_projection_records_column_shares_and_replays_mount_bytes() -> Result<()> {
    use crate::shared::{experts::rtx::dense, memory::device::Device};
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    fixture.immediate_copies(&library)?;
    for device in 0..2 { fixture.configure_memory(&library, device, 1 << 30)?; }
    for shape in [[8, 256], [4096, 8448]] {
    let checkpoint = checkpoint(&fixture, &[("weight", DType::Bf16, &shape)])?;
    let path = checkpoint.snapshot.join("weights.bin");
    let payload: Vec<_> = (0..shape[0] * shape[1] * 2).map(|i| (i % 251) as u8).collect();
    std::fs::write(&path, &payload)?;
    let formats = BTreeMap::new();
    for ranks in [1, 2] {
        let timing = crate::shared::memory::load_timing::Split::start();
        let mounted = loader(&library, &checkpoint, &formats).o_proj("weight", ranks)?;
        assert_eq!(timing.read_bytes(), payload.len() as u64, "each source byte is read once");
        std::fs::write(&path, [])?;
        let load = std::rc::Rc::new(dense::DenseLoad::push(&library));
        let planning = MimoLoader { dense: load.clone(), ..loader(&library, &checkpoint, &formats) };
        let (pushed, mut records) = dense::recording(|| planning.o_proj("weight", ranks))?;
        records.extend(load.sink.as_ref().unwrap().take_all());
        let per = cuteafd_loader::shared::dense_read::ReadBatch::new(
            shape[0], shape[1] * 2, shape[1] * 2 / ranks, dense::STEP_BYTES)?.rows;
        assert_eq!(records.operands.len(), ranks * shape[0].div_ceil(per));
        assert!(records.steps.is_empty() && records.finishes.is_empty());
        std::fs::write(&path, &payload)?;
        let sockets = tempfile::tempdir_in(fixture.directory())?;
        for rank in 0..ranks {
            crate::shared::experts::rtx::push::tests::receive_pushed(
                Device { library: &library, id: rank as i32 }, None, &checkpoint.snapshot,
                sockets.path(), &records, None, 0, rank)?;
            let expected: Vec<_> = payload.chunks_exact(shape[1] * 2).flat_map(|row|
                row[rank * shape[1] * 2 / ranks..(rank + 1) * shape[1] * 2 / ranks].iter().copied()).collect();
            for allocation in [&mounted[rank], &pushed[rank]] {
                let mut actual = vec![0; expected.len()];
                library.copy_d2h(&mut actual, allocation.buffer)?;
                assert_eq!(actual, expected);
            }
        }
    }
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn successful_loaders_reclaim_storage_and_unload_normally() -> Result<()> {
    for path in 0..2 {
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
            _ => drop(loader.fp8_projection("weight", 1, false)?),
        }
        drop(preceding_layer);
        drop(preceding_pin);
        drop(loader);
        drop(library);
        let events = fixture.events()?;
        assert_eq!(count(&events, 'D'), count(&events, 'd'), "{events}");
        assert!(events.find('S') < events.find('d'), "{events}");
        assert!(events.ends_with("FU"), "{events}");
        assert!(!fixture.resident()?);
    }
    Ok(())
}
