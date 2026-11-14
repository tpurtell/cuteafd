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
#[ignore = "requires EXL3 target+MTP checkpoint, coordinator packages and one CUDA device"]
fn independently_backed_target_and_draft_share_serial_scratch_exactly() -> Result<()> {
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    library.cuda_set_device(0)?;
    let catalog = cuteafd_loader::read_expert_catalog(Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
    let geometry = catalog.routed_experts().geometry()?;
    cuteafd_core::set_expert_geometry(geometry).map_err(|_| anyhow::anyhow!("expert geometry already selected"))?;
    let root = PathBuf::from(std::env::var("CUTEAFD_EXL3_AOT")?);
    let rows = 64usize;
    let paths = cuteafd_core::Exl3RowPolicy::for_geometry(geometry).capacities(rows)?
        .into_iter().map(|capacity| root.join(format!("m{capacity}"))).collect::<Vec<_>>();
    let device = crate::shared::memory::device::Device { library: &library, id: 0 };
    let stream = crate::shared::memory::device::Stream::new(device)?;
    let target = Rc::new(vec![Exl3Weights::load(&library, &catalog, ExpertLayer::BackboneFull { layer: 0 }, usize::MAX)?]);
    let draft = Rc::new(vec![Exl3Weights::load(&library, &catalog, ExpertLayer::Dspark { stage: 0 }, usize::MAX)?]);
    let output_bytes = rows * geometry.hidden as usize * 4;
    let arena = Exl3Workspace::serial(&library, &paths, Exl3Epilogue::Sum, stream.raw, output_bytes)?;
    let output = arena.serial_output(&library, stream.raw, output_bytes)?;
    let mut private = Vec::new();
    let mut shared = Vec::new();
    for weights in [target, draft] {
        // SAFETY: these trusted package bindings run exclusively on stream below.
        private.push(unsafe { Exl3Execution::with_input_format(&library, weights.clone(), paths.last().unwrap(),
            Exl3InputFormat::Fp8K32, Exl3Epilogue::Sum)? });
        // SAFETY: both independently backed sets share the arena on one serialized stream.
        shared.push(unsafe { Exl3Execution::with_shared_workspace(&library, weights, paths.last().unwrap(),
            Exl3InputFormat::Fp8K32, Exl3Epilogue::Sum, Some(arena.clone()))? });
    }
    let (hidden, topk) = (geometry.hidden as usize, geometry.topk as usize);
    let pitch = hidden + hidden / 32;
    let input = [DeviceAllocation::new(&library, rows * pitch, crate::shared::memory::TEST)?,
        DeviceAllocation::new(&library, rows * topk * 4, crate::shared::memory::TEST)?,
        DeviceAllocation::new(&library, rows * topk * 4, crate::shared::memory::TEST)?];
    let mut wire = vec![0u8; rows * pitch];
    for (row, values) in wire.chunks_exact_mut(pitch).enumerate() {
        for (column, value) in values[..hidden].iter_mut().enumerate() { *value = 0x30 + ((row + column) % 16) as u8; }
        values[hidden..].fill(127);
    }
    library.copy_h2d(input[0].buffer, &wire)?;
    library.copy_h2d(input[1].buffer, &(0..rows * topk).flat_map(|i| ((i % topk) as i32).to_ne_bytes()).collect::<Vec<_>>())?;
    library.copy_h2d(input[2].buffer, &(0..rows * topk).flat_map(|_| (1.0f32 / topk as f32).to_ne_bytes()).collect::<Vec<_>>())?;
    let buffers = || input.each_ref().map(|input| input.buffer);
    let read = |buffer: CuteafdDeviceBuffer| -> Result<Vec<u8>> {
        let mut bytes = vec![0; rows * hidden * 4];
        library.copy_d2h(&mut bytes, CuteafdDeviceBuffer { bytes: bytes.len(), ..buffer })?;
        Ok(bytes)
    };
    let mut expected = Vec::new();
    for binding in &mut private {
        // SAFETY: live distinct input/output, one stream and retained weight owners.
        let output = unsafe { binding.launch(buffers(), rows, stream.raw)? };
        stream.drain()?;
        expected.push(read(output)?);
    }
    for index in [0usize, 1, 0, 1] {
        // SAFETY: target/draft never overlap; output is one retained serial backing.
        unsafe { shared[index].launch_layer_into(0, buffers(), rows, stream.raw, output.buffer)?; }
        stream.drain()?;
        ensure!(read(output.buffer)? == expected[index], "shared target/draft output differs for binding {index}");
    }
    Ok(())
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
        let inputs = [DeviceAllocation::new(&library, 4096*pitch, crate::shared::memory::TEST)?,
            DeviceAllocation::new(&library, 4096*k*4, crate::shared::memory::TEST)?, DeviceAllocation::new(&library, 4096*k*4, crate::shared::memory::TEST)?];
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
                exec: unsafe { library.cuda_graph_end_capture_for_test(stream.raw)? }, stream: stream.raw };
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
    std::fs::write(directory.join("lut"), [0u8; 16])?;
    let lock: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"), "/../../../third_party/sparkinfer.lock.json")))?;
    let mut buffers = serde_json::Map::new();
    for (name, bytes, dtype) in [("output", 5120*4, "torch.float32"), ("fc2", 16, "torch.float32"),
        ("packed_route_indices", 16, "torch.int32"), ("block_expert_ids", 16, "torch.int32"),
        ("packed_route_count", 16, "torch.int32"), ("expert_offsets", 16, "torch.int32"),
        ("expert_counts", 16, "torch.int32")] {
        buffers.insert(name.into(), serde_json::json!({"bytes":bytes,"dtype":dtype,"allocation":name,"zero_on_create":false}));
    }
    let meta = serde_json::json!({
        "schema":"cuteafd.v41-exl3-aot.v1", "output_dtype":"fp32", "sparkinfer_revision":lock["revision"],
        "hidden":5120,"intermediate":1152,"experts":6,"capacity":1,"top_k":6,"bits":[3,4],
        "swiglu_limit":cuteafd_core::expert_geometry().swiglu_limit(),"direct":false,"sms":1,"blocks_per_sm":1,
        "buffers":buffers,"objects":[
            {"label":"v41_exl3_core","pointer_slots":["rotation_input_ptr"],"scalar_slots":["active_m"]},
            {"label":"v41_exl3_sum","pointer_slots":["output_ptr"],"scalar_slots":["active_m"]}],
        "trellis_lut":{"file":"lut","bytes":16,"sha256":format!("{:x}", Sha256::digest([0u8;16]))},
        "route_preparation":{"manifest":"routes.json","sha256":format!("{:x}",Sha256::digest(b"{}"))}
    });
    std::fs::write(directory.join("v41_exl3.json"), serde_json::to_vec(&meta)?)?;
    library.cuda_set_device(0)?;
    let mut lanes = Vec::new();
    let inventory = crate::shared::inventory::RuntimeInventory::measure(&library, &[1], crate::shared::inventory::InventoryPhase::PreWeight, |_| {
        for _ in 0..2 {
            // SAFETY: CPU-only fixture implements the trusted module ABI.
            lanes.push(unsafe { Exl3Modules::load(&library, directory, Exl3InputFormat::Fp8K32)? });
        }
        Ok(())
    })?;
    assert_eq!(library.cuda_get_device()?, 0);
    assert_eq!(inventory.gpus[0].module_bytes, 2*(4096+1024+512));
    assert_eq!(fixture.events()?, "KRWKRW", "inventory must retain every context");
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
    let arena = DeviceAllocation::new(&library, 1024, crate::shared::memory::TEST)?;
    let weights = Rc::new(vec![Exl3Weights {
        buffers: specs.iter().map(|s| WeightBuffer { buffer: CuteafdDeviceBuffer { bytes:s.bytes, ..arena.buffer } }).collect(),
        _arena: crate::shared::experts::exl3::ArenaOwner::Mount(arena),
        layout: V41Exl3Residency { layer:V41Exl3Layer::Backbone(0),world:2,rank:1,
            intermediate_start:1152,experts:6,intermediate:1152,tiers:vec![3,4],projection_counts:vec![[0;3];2],unit_map:None,buffers:specs,loads:vec![] },
        budget: ExpertLoadBudget { resident_bytes:1024,device_staging_bytes:0,pinned_host_bytes:0,read_scratch_bytes:0 },
    }]);
    let mut executions = Vec::new();
    for modules in lanes {
        // SAFETY: private scratch, no queued work or graphs in the CPU fixture.
        executions.push(unsafe { Exl3Execution::with_modules(&library, weights.clone(), modules,
            Exl3Epilogue::Sum, None)? });
    }
    assert_eq!(fixture.events()?.chars().filter(|c| "KRW".contains(*c)).collect::<String>(), "KRWKRW",
        "execution must consume the measured handles, never reload");
    // Dropping retained native contexts restores their own device even from a peer.
    library.cuda_set_device(0)?;
    drop(executions);
    assert_eq!(library.cuda_get_device()?, 0);
    assert_eq!(fixture.events()?.chars().filter(|c| "krw".contains(*c)).collect::<String>(), "krwkrw");
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
