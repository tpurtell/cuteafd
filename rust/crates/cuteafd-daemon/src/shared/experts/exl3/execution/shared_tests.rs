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
    let catalog = cuteafd_loader::read_official_v41_catalog(
        cuteafd_loader::OFFICIAL_V41_MODEL_ID, Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
    let root = PathBuf::from(std::env::var("CUTEAFD_EXL3_AOT")?);
    let tp1 = std::env::var_os("CUTEAFD_EXL3_TEST_TP1").is_some();
    let capacities = [1usize, 16, 80, 256, 1024, 4096];
    let directories: Vec<_> = capacities.iter().map(|c| root.join(format!("m{c}"))).collect();
    let format = Exl3InputFormat::Fp8K32;
    for device in 0..2 {
        library.cuda_set_device(device)?;
        let weights = Rc::new(vec![Exl3Weights::load(&library, &catalog,
            if tp1 { ExpertLayer::BackboneFull { layer: 0 } } else { ExpertLayer::BackboneTp2 { layer: 0, rank: device as usize } }, 7_000_000_000)?]);
        let arena = Exl3Workspace::new(&library, &directories)?;
        let mut states = directories.iter().map(|path| unsafe {
            Exl3Execution::with_shared_workspace(&library, weights.clone(), path, format, Some(arena.clone()))
        }).collect::<Result<Vec<_>>>()?;
        let actual_bytes = arena.allocations.values().map(|a| a.buffer.bytes).sum::<usize>()
            + states.iter().map(|s| s.workspace_bytes()).sum::<usize>();
        let planned = Exl3Workspace::plan(&directories, format)?;
        ensure!(actual_bytes == planned, "shared allocation budget mismatch");
        let old = directories.iter().map(|p| Exl3Execution::plan(p, format)).collect::<Result<Vec<_>>>()?
            .into_iter().sum::<usize>();
        ensure!(planned < old, "workspace sharing did not reduce storage");
        let inputs = [DeviceAllocation::new(&library, 4096*5280)?,
            DeviceAllocation::new(&library, 4096*24)?, DeviceAllocation::new(&library, 4096*24)?];
        let mut wire = vec![0;4096*5280];
        for (row, bytes) in wire.chunks_exact_mut(5280).enumerate() {
            for (col, value) in bytes[..5120].iter_mut().enumerate() { *value = 0x30 + ((row+col)%16) as u8; }
            bytes[5120..].fill(127);
        }
        library.copy_h2d(inputs[0].buffer, &wire)?;
        library.copy_h2d(inputs[1].buffer, &(0..4096*6i32).flat_map(|i| (i%6).to_ne_bytes()).collect::<Vec<_>>())?;
        library.copy_h2d(inputs[2].buffer, &(0..4096*6).flat_map(|_| (1f32/6.).to_ne_bytes()).collect::<Vec<_>>())?;
        let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let read = |buffer: CuteafdDeviceBuffer| -> Result<Vec<u8>> {
            let mut bytes = vec![0; buffer.bytes]; library.copy_d2h(&mut bytes, buffer)?; Ok(bytes)
        };
        let input_buffers = || std::array::from_fn(|i| inputs[i].buffer);
        let mut graphs = Vec::new();
        for (i, &capacity) in capacities.iter().enumerate() {
            let mut private = unsafe { Exl3Execution::with_input_format(&library, weights.clone(), &directories[i], format)? };
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
            for row in zero.chunks_exact_mut(5280) { row[..5120].fill(0); }
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
    use cuteafd_loader::{V41Exl3Layer, V41Exl3Partition, V41Exl3Residency, V41Exl3ResidentBuffer};
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
    let inventory = crate::shared::inventory::RuntimeInventory::measure(&library, &[1], |_| {
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
    let specs: Vec<_> = names.into_iter().map(|name| V41Exl3ResidentBuffer {
        bytes: if name == "descriptor_map" {3*6*2*4} else {16}, name, initial_words: vec![] }).collect();
    let arena = DeviceAllocation::new(&library, 1024)?;
    let weights = Rc::new(vec![Exl3Weights {
        buffers: specs.iter().map(|s| WeightBuffer { buffer: CuteafdDeviceBuffer { bytes:s.bytes, ..arena.buffer } }).collect(),
        _arena: arena,
        layout: V41Exl3Residency { layer:V41Exl3Layer::Backbone(0),world:2,rank:1,layout:V41Exl3Partition::Disjoint,
            intermediate_start:1152,experts:6,intermediate:1152,tiers:vec![3,4],projection_counts:vec![[0;3];2],buffers:specs,loads:vec![] },
        budget: ExpertLoadBudget { resident_bytes:1024,device_staging_bytes:0,pinned_host_bytes:0,read_scratch_bytes:0 },
    }]);
    let mut executions = Vec::new();
    for modules in lanes {
        // SAFETY: private scratch, no queued work or graphs in the CPU fixture.
        executions.push(unsafe { Exl3Execution::with_modules(&library, weights.clone(), modules, None)? });
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
    let error = crate::shared::inventory::RuntimeInventory::measure(&library, &[1], |_| {
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
