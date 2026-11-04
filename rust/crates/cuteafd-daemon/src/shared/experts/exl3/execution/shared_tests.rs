use super::*;
use crate::shared::experts::layer::ExpertLayer;
use crate::shared::memory::LoadStream;

struct Graph<'a> { library: &'a NativeLibrary, exec: *mut c_void, stream: *mut c_void }
impl Drop for Graph<'_> {
    fn drop(&mut self) {
        unsafe {
            let _ = self.library.cuda_stream_synchronize(self.stream);
            let _ = self.library.cuda_graph_exec_destroy(self.exec);
        }
    }
}

#[test]
#[ignore = "requires EXL3 snapshot, TP2 package, native library and two CUDA devices"]
fn shared_capacity_scratch_matches_private_and_graph_replay() -> Result<()> {
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    let catalog = cuteafd_loader::read_expert_catalog(Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
    let geometry = catalog.routed_experts().geometry()?;
    let h = geometry.hidden as usize;
    let k = geometry.topk as usize;
    let pitch = h + h / 32;
    let root = PathBuf::from(std::env::var("CUTEAFD_EXL3_AOT")?);
    let tp1 = std::env::var_os("CUTEAFD_EXL3_TEST_TP1").is_some();
    let capacities = [1usize, 16, 80, 256, 1024, 4096];
    let directories: Vec<_> = capacities.iter().map(|c| root.join(format!("m{c}"))).collect();
    let format = Exl3InputFormat::Fp8K32;
    for device in 0..2 {
        library.cuda_set_device(device)?;
        let weights = Rc::new(vec![Exl3Weights::load(&library, &catalog,
            if tp1 { ExpertLayer::BackboneFull { layer: 0 } } else { ExpertLayer::BackboneTp2 { layer: 0, rank: device as usize } }, 7_000_000_000)?]);
        let arena = Exl3Workspace::new(&library, &directories, Exl3Epilogue::Sum)?;
        let mut states = directories.iter().map(|path| unsafe {
            Exl3Execution::with_shared_workspace(&library, weights.clone(), path, format,
                Exl3Epilogue::Sum, Some(arena.clone()))
        }).collect::<Result<Vec<_>>>()?;
        let actual_bytes = arena.allocations.values().map(|a| a.buffer.bytes).sum::<usize>()
            + states.iter().map(|s| s.workspace_bytes()).sum::<usize>();
        let planned = Exl3Workspace::plan(&directories, format, Exl3Epilogue::Sum)?;
        ensure!(actual_bytes == planned, "shared allocation budget mismatch");
        let old = directories.iter().map(|p| Exl3Execution::plan(p, format, Exl3Epilogue::Sum)).collect::<Result<Vec<_>>>()?
            .into_iter().sum::<usize>();
        ensure!(planned < old, "workspace sharing did not reduce storage");
        let inputs = [DeviceAllocation::new(&library, 4096*pitch)?,
            DeviceAllocation::new(&library, 4096*k*4)?, DeviceAllocation::new(&library, 4096*k*4)?];
        let mut wire = vec![0;4096*pitch];
        for (row, bytes) in wire.chunks_exact_mut(pitch).enumerate() {
            for (col, value) in bytes[..h].iter_mut().enumerate() { *value = 0x30 + ((row+col)%16) as u8; }
            bytes[h..].fill(127);
        }
        library.copy_h2d(inputs[0].buffer, &wire)?;
        library.copy_h2d(inputs[1].buffer, &(0..(4096*k) as i32).flat_map(|i| (i%k as i32).to_ne_bytes()).collect::<Vec<_>>())?;
        library.copy_h2d(inputs[2].buffer, &(0..4096*k).flat_map(|_| (1f32/k as f32).to_ne_bytes()).collect::<Vec<_>>())?;
        let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let read = |buffer: CuteafdDeviceBuffer| -> Result<Vec<u8>> {
            let mut bytes = vec![0; buffer.bytes]; library.copy_d2h(&mut bytes, buffer)?; Ok(bytes)
        };
        let input_buffers = || std::array::from_fn(|i| inputs[i].buffer);
        let mut graphs = Vec::new();
        for (i, &capacity) in capacities.iter().enumerate() {
            let mut private = unsafe { Exl3Execution::with_input_format(&library, weights.clone(),
                &directories[i], format, Exl3Epilogue::Sum)? };
            let output = unsafe { private.launch(input_buffers(), capacity, stream.raw)? };
            unsafe { library.cuda_stream_synchronize(stream.raw)?; }
            let expected = read(output)?;
            ensure!(expected.chunks_exact(4).any(|b| f32::from_ne_bytes(b.try_into().unwrap()) != 0.), "zero reference");
            let actual = unsafe { states[i].launch(input_buffers(), capacity, stream.raw)? };
            unsafe { library.cuda_stream_synchronize(stream.raw)?; }
            ensure!(read(actual)? == expected, "private/shared mismatch at {capacity}");
            unsafe { library.cuda_graph_begin_capture(stream.raw)?; }
            let captured = unsafe { states[i].launch(input_buffers(), capacity, stream.raw) };
            let graph = Graph { library: &library,
                exec: unsafe { library.cuda_graph_end_capture(stream.raw)? }, stream: stream.raw };
            graphs.push((graph, captured?, expected));
        }
        // Replay every capacity after larger and smaller kernels have overwritten
        // the arena, with both zeroed and restored inputs. Counters stay private.
        for i in (0..capacities.len()).rev().chain(0..capacities.len()) {
            let (graph, output, expected) = &graphs[i];
            let mut zero = wire.clone();
            for row in zero.chunks_exact_mut(pitch) { row[..h].fill(0); }
            library.copy_h2d(inputs[0].buffer, &zero)?;
            unsafe { library.cuda_graph_launch(graph.exec, stream.raw)?; library.cuda_stream_synchronize(stream.raw)?; }
            ensure!(read(*output)?.chunks_exact(4).all(|b| f32::from_ne_bytes(b.try_into().unwrap()) == 0.), "stale graph input");
            library.copy_h2d(inputs[0].buffer, &wire)?;
            unsafe { library.cuda_graph_launch(graph.exec, stream.raw)?; library.cuda_stream_synchronize(stream.raw)?; }
            ensure!(read(*output)? == *expected, "shared graph mismatch at {}", capacities[i]);
        }
        eprintln!("GPU {device}: shared={planned} private={old} saved={} bytes; six capacities, changed-input graph replay exact", old-planned);
    }
    Ok(())
}

fn verify_workspace_receipts(states: &[Exl3Execution<'_>], manifests: &[serde_json::Value],
    location: cuteafd_loader::placement::planner::Location, physical_device: i32) -> Result<()> {
    use cuteafd_loader::placement::planner::{self, *};
    let lifetime = Lifetime { first: Phase::Warmup, last: Phase::Growth };
    let layout = cuteafd_loader::serving_capacity::Exl3WorkspaceLayout::from_manifests(
        manifests, true, Exl3Epilogue::Sum)?;
    let facts = Facts {phase_order: cuteafd_loader::placement::planner::PhaseOrder::TARGET_THEN_DRAFT,
        allocations: layout.allocations("expert.test", location, lifetime),
        deferred: vec![], sequential: vec![], reservations: vec![], allowances: vec![],
    };
    let budgets = vec![Budget {
        id: "fixture physical device".into(), locations: vec![location],
        physical_bytes: 1 << 30, logical_bytes: 1 << 30,
        baseline: Baseline::Forecast { occupied: 0, source: "CPU fixture".into(), assumed: false },
        headroom: 0,
    }];
    let phases = planner::evaluate(&facts, &budgets)?;
    let plan = ResolvedPlan {
        request: Request { context: 64, pool: Pin::Exact(64), onboard: Pin::Auto,
            prefill: Pin::Auto, spark_reply_format: None, operation: Operation::Serving },
        decisions: Decisions { spark_reply_format: None, attention: vec![], onboard_layers: vec![vec![], vec![]],
            expert_homes: Default::default(), embedding: None,
            prefill: PrefillShape { rows: 64, waves: 1 }, lanes: 1 },
        pool_tokens: 64, budgets, facts, phases,
    };
    let receipts = Exl3Execution::workspace_receipts(states, "expert.test", location, physical_device, lifetime)?;
    planner::reconcile(&plan, Phase::Warmup, &receipts)?;
    assert!(receipts.iter().all(|receipt| receipt.allocation.location == location));
    assert!(Exl3Execution::workspace_receipts(states, "expert.test", location, -1, lifetime).is_err());
    assert!(Exl3Execution::workspace_receipts(states, "expert.test", location, 1-physical_device, lifetime).is_err());
    assert!(Exl3Execution::workspace_receipts(states, "", location, physical_device, lifetime).is_err());
    assert!(Exl3Execution::workspace_receipts(&[], "expert.test", location, physical_device, lifetime).is_err());
    for invalid in [Location::HOST, Location::PINNED,
        Location { device: Device::Host, domain: Domain::Device },
        Location { device: Device::Spark(0), domain: Domain::Managed },
        Location { device: Device::Coordinator(0), domain: Domain::Managed }] {
        assert!(Exl3Execution::workspace_receipts(states, "expert.test", invalid, physical_device, lifetime).is_err());
    }
    let physical: Vec<_> = receipts.iter().enumerate()
        .filter(|(_, r)| r.allocation.view.is_none()).map(|(i, _)| i).collect();
    assert_eq!(physical.iter().map(|&i| receipts[i].allocation.bytes).sum::<u64>(), layout.bytes()?);
    assert!(receipts.iter().any(|r| r.allocation.id.ends_with(".trellis_lut") && r.allocation.bytes == 16));
    assert!(receipts.iter().any(|r| r.allocation.id.ends_with(".storage.epoch") && r.allocation.bytes == 16));
    for negative in 0..9 {
        let mut bad = receipts.clone();
        match negative {
            0 => { bad.pop(); },
            1 => { let mut extra = bad[physical[0]].clone(); extra.allocation.id.push_str(".unexpected"); bad.push(extra); },
            2 => bad[physical[0]].allocation.bytes += 1,
            3 => { bad[physical[0]].allocation.bytes += 1; bad[physical[1]].allocation.bytes -= 1; },
            4 => bad[physical[0]].allocation.location = if location == Location::coordinator(0) {
                Location::coordinator(1)
            } else { Location::coordinator(0) },
            5 => bad[physical[0]].address = 0,
            6 => bad[physical[1]].address = bad[physical[0]].address,
            7 => bad.iter_mut().find(|r| r.allocation.view.is_some()).unwrap().address += 1,
            8 => bad.iter_mut().find(|r| r.allocation.view.is_some()).unwrap()
                .allocation.view.as_mut().unwrap().backing.push_str(".wrong"),
            _ => unreachable!(),
        }
        assert!(planner::reconcile(&plan, Phase::Warmup, &bad).is_err(), "negative {negative}");
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn retained_modules_are_measured_once_and_consumed_without_reload() -> Result<()> {
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::{V41Exl3Layer, V41Exl3Residency, V41Exl3ResidentBuffer};
    use crate::shared::experts::exl3::WeightBuffer;
    use crate::shared::experts::layer::ExpertLoadBudget;
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    fixture.configure_memory(&library, 1, 1 << 30)?;
    let directory = fixture.directory();
    let native = directory.join("liblifetime_fixture.so");
    for name in ["libcuteafd_exl3.so", "libv41_exl3_routes.so"] {
        std::os::unix::fs::symlink(&native, directory.join(name))?;
    }
    std::fs::write(directory.join("routes.json"), b"{}")?;
    std::fs::write(directory.join("lut"), [0u8; 7])?;
    let lock: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"), "/../../../third_party/sparkinfer.lock.json")))?;
    let mut buffers = serde_json::Map::new();
    for (name, bytes, dtype) in [("output", 5120*4, "torch.float32"), ("fc2", 16, "torch.float32"),
        ("packed_route_indices", 16, "torch.int32"), ("block_expert_ids", 16, "torch.int32"),
        ("packed_route_count", 16, "torch.int32"), ("expert_offsets", 16, "torch.int32"),
        ("expert_counts", 16, "torch.int32")] {
        buffers.insert(name.into(), serde_json::json!({"bytes":bytes,"dtype":dtype,"allocation":name,"zero_on_create":false}));
    }
    buffers.insert("epoch".into(), serde_json::json!({
        "bytes":0,"dtype":"torch.int32","allocation":"epoch","zero_on_create":true}));
    buffers.insert("alias".into(), serde_json::json!({
        "bytes":8,"dtype":"torch.float32","allocation":"fc2","zero_on_create":false}));
    buffers.insert("epoch_alias".into(), serde_json::json!({
        "bytes":8,"dtype":"torch.int32","allocation":"epoch","zero_on_create":false}));
    let meta = serde_json::json!({
        "schema":"cuteafd.v41-exl3-aot.v1", "output_dtype":"fp32", "sparkinfer_revision":lock["revision"],
        "hidden":5120,"intermediate":1152,"experts":6,"capacity":1,"top_k":6,"bits":[3,4],
        "swiglu_limit":cuteafd_core::expert_geometry().swiglu_limit(),"direct":false,"sms":1,"blocks_per_sm":1,
        "buffers":buffers,"objects":[
            {"label":"v41_exl3_core","pointer_slots":["rotation_input_ptr"],"scalar_slots":["active_m"]},
            {"label":"v41_exl3_sum","pointer_slots":["output_ptr"],"scalar_slots":["active_m"]}],
        "trellis_lut":{"file":"lut","bytes":7,"sha256":format!("{:x}", Sha256::digest([0u8;7]))},
        "route_preparation":{"manifest":"routes.json","sha256":format!("{:x}",Sha256::digest(b"{}"))}
    });
    std::fs::write(directory.join("v41_exl3.json"), serde_json::to_vec(&meta)?)?;
    let wide_directory = directory.join("m16");
    std::fs::create_dir(&wide_directory)?;
    for name in ["libcuteafd_exl3.so", "libv41_exl3_routes.so", "routes.json", "lut"] {
        std::os::unix::fs::symlink(directory.join(name), wide_directory.join(name))?;
    }
    let mut wide = meta.clone();
    wide["capacity"] = serde_json::json!(16);
    wide["buffers"]["output"]["bytes"] = serde_json::json!(16*5120*4);
    wide["buffers"]["fc2"]["bytes"] = serde_json::json!(128);
    wide["input_format"] = serde_json::json!("e4m3_k32");
    std::fs::write(wide_directory.join("v41_exl3.json"), serde_json::to_vec(&wide)?)?;
    library.cuda_set_device(0)?;
    let mut lanes = Vec::new();
    let inventory = crate::shared::inventory::RuntimeInventory::measure(&library, &[1], crate::shared::inventory::InventoryPhase::PreWeight, |_| {
        for (path, capacity) in [(directory, 1), (wide_directory.as_path(), 16)] {
            fixture.configure_exl3_capacity(&library, capacity)?;
            // SAFETY: CPU-only fixture implements the trusted module ABI.
            lanes.push(unsafe { Exl3Modules::load(&library, path, Exl3InputFormat::Fp8K32)? });
        }
        Ok(())
    })?;
    fixture.configure_exl3_capacity(&library, 1)?;
    assert_eq!(library.cuda_get_device()?, 0);
    assert_eq!(inventory.gpus[0].module_bytes, 2*(4096+1024)+512);
    assert_eq!(fixture.events()?, "KRWKR", "inventory must retain every context");
    library.cuda_set_device(1)?;
    let mut names: Vec<String> = ["descriptor_map", "global_to_combined", "intermediate_rotations", "gate_suh", "up_suh", "down_svh", "dummy_scales", "unit_scales"]
        .into_iter().map(str::to_owned).collect();
    for tier in 0..2 { for suffix in ["w13", "w2"] { names.push(format!("tier{tier}_{suffix}")); } }
    // A resident descriptor map carries its initial words, as every planned
    // residency does (legacy 3-row descriptors: rows x experts x tiers).
    let specs: Vec<_> = names.into_iter().map(|name| {
        let descriptor = name == "descriptor_map";
        V41Exl3ResidentBuffer { bytes: if descriptor {3*6*2*4} else {16},
            initial_words: if descriptor { vec![0; 3*6*2] } else { vec![] }, name }
    }).collect();
    let layout = V41Exl3Residency { layer:V41Exl3Layer::Backbone(0),world:2,rank:0,
        intermediate_start:0,experts:6,intermediate:1152,tiers:vec![3,4],projection_counts:vec![[0;3];2],
        unit_map:None,buffers:specs,loads:vec![] };
    let (offsets, bytes) = layout.device_arena_layout()?;
    let arena = DeviceAllocation::new(&library, bytes)?;
    let buffers = layout.buffers.iter().zip(offsets).map(|(spec, offset)| {
        // SAFETY: every aligned view lies in this header-derived live arena.
        let ptr = unsafe { arena.buffer.ptr.cast::<u8>().add(offset).cast() };
        WeightBuffer { buffer: CuteafdDeviceBuffer { ptr, bytes:spec.bytes.max(16), ..arena.buffer } }
    }).collect();
    let weights = Rc::new(vec![Exl3Weights { buffers,
        _arena: crate::shared::experts::exl3::ArenaOwner::Mount(arena), layout,
        budget: ExpertLoadBudget { resident_bytes:bytes,device_staging_bytes:0,pinned_host_bytes:0,read_scratch_bytes:0 },
    }]);
    let workspace = Exl3Workspace::new(&library, &[directory.to_owned(), wide_directory], Exl3Epilogue::Sum)?;
    let mut executions = Vec::new();
    for modules in lanes {
        // SAFETY: allocation-only CPU fixture; no queued work, launches or graphs.
        executions.push(unsafe { Exl3Execution::with_modules(&library, weights.clone(), modules,
            Exl3Epilogue::Sum, Some(workspace.clone()))? });
    }
    use cuteafd_loader::placement::planner::{Device, Domain, Lifetime, Location};
    let manifests = [meta.clone(), wide];
    for location in [Location::coordinator(0), Location::coordinator(1),
        Location { device: Device::Spark(3), domain: Domain::Device }] {
        verify_workspace_receipts(&executions, &manifests, location, 1)?;
    }
    executions[0].device = 0;
    assert!(Exl3Execution::workspace_receipts(&executions, "expert.test", Location::coordinator(0),
        1, Lifetime::RESIDENT).is_err());
    executions[0].device = 1;
    let binding = executions[0].workspace_bindings.values_mut().next().unwrap();
    let original_device = binding.device_id;
    binding.device_id = 0;
    assert!(Exl3Execution::workspace_receipts(&executions, "expert.test", Location::coordinator(0),
        1, Lifetime::RESIDENT).is_err());
    executions[0].workspace_bindings.values_mut().next().unwrap().device_id = original_device;
    for execution in &executions {
        execution.validate_retained_weights(&weights, 1)?;
        assert!(execution.validate_retained_weights(&Rc::new(Vec::new()), 1).is_err());
    }
    let mut wrong_owner = Rc::new(Vec::new());
    std::mem::swap(&mut executions[0]._weights, &mut wrong_owner);
    assert!(executions[0].validate_retained_weights(&weights, 1).is_err());
    std::mem::swap(&mut executions[0]._weights, &mut wrong_owner);
    let original_map = executions[0].layers[0].expert_map;
    executions[0].layers[0].expert_map.ptr = std::ptr::null_mut();
    assert!(executions[0].validate_retained_weights(&weights, 1).is_err());
    executions[0].layers[0].expert_map = original_map;
    executions = crate::shared::experts::rtx::exl3::tests::verify_retained_rtx_receipts(
        crate::shared::memory::device::Device { library: &library, id: 1 },
        weights.clone(), executions, &manifests)?;
    assert_eq!(fixture.events()?.chars().filter(|c| "KRW".contains(*c)).collect::<String>(), "KRWKR",
        "execution must consume the measured handles, never reload");
    // Dropping retained native contexts restores their own device even from a peer.
    library.cuda_set_device(0)?;
    drop(executions);
    library.cuda_set_device(1)?;
    drop(workspace);
    library.cuda_set_device(0)?;
    assert_eq!(library.cuda_get_device()?, 0);
    assert_eq!(fixture.events()?.chars().filter(|c| "krw".contains(*c)).collect::<String>(), "krwkr");
    library.cuda_set_device(1)?;
    drop(weights);
    assert_eq!(library.cuda_memory_info()?, (1 << 30, 1 << 30));

    // An admission that never constructs execution still owns device-correct cleanup.
    let before = fixture.events()?.len();
    // SAFETY: CPU-only fixture implements the trusted module ABI.
    let unused = unsafe { Exl3Modules::load(&library, directory, Exl3InputFormat::Fp8K32)? };
    library.cuda_set_device(0)?;
    drop(unused);
    assert_eq!(library.cuda_get_device()?, 0);
    assert_eq!(&fixture.events()?[before..], "KRWkrw");

    // A later module failure releases the already loaded core before inventory exits.
    let mut invalid = meta;
    invalid["route_preparation"]["sha256"] = serde_json::json!("invalid");
    std::fs::write(directory.join("v41_exl3.json"), serde_json::to_vec(&invalid)?)?;
    let before = fixture.events()?.len();
    let error = crate::shared::inventory::RuntimeInventory::measure(&library, &[1], crate::shared::inventory::InventoryPhase::PreWeight, |_| {
        // SAFETY: the trusted fixture intentionally carries a failing route hash.
        let _modules = unsafe { Exl3Modules::load(&library, directory, Exl3InputFormat::Fp8K32)? };
        Ok(())
    }).expect_err("invalid route hash must fail admission");
    assert!(format!("{error:#}").contains("EXL3 route export hash mismatch"));
    assert_eq!(library.cuda_get_device()?, 0);
    assert_eq!(&fixture.events()?[before..], "Kk");
    library.cuda_set_device(1)?;
    assert_eq!(library.cuda_memory_info()?, (1 << 30, 1 << 30));
    library.cuda_set_device(0)?;
    Ok(())
}
