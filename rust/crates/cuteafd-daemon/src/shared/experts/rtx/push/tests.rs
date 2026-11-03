use super::*;
use cuteafd_loader::push_load::builder::Builder;
use crate::shared::memory::device::Device;
use anyhow::Context;
use cuteafd_ffi::{native_library_lifetime_fixture::Fixture, push_ring::DEFAULT_SLOT_BYTES};
use cuteafd_loader::{plan::testing::{qwen4_config, qwen4_exl3, write_snapshot},
    push_load::{proto::{Event, Request}, slot, Listener, PushClient}};
use std::{fs::File, net::TcpStream, os::unix::fs::FileExt, time::Duration};

#[test]
fn whole_binding_requires_exact_selected_inventory_and_admission() -> Result<()> {
    use crate::shared::experts::layer::ExpertLayer::BackboneFull;
    let expected = [(BackboneFull { layer: 0 }, 256), (BackboneFull { layer: 1 }, 512)];
    validate_whole_inventory(&expected, &expected, 768)?;
    assert!(validate_whole_inventory(&expected, &expected, 767).is_err());
    assert!(validate_whole_inventory(&expected[..1], &expected, 768).is_err());
    assert!(validate_whole_inventory(&[expected[1], expected[0]], &expected, 768).is_err());
    assert!(validate_whole_inventory(&[expected[0], expected[0]], &expected, 768).is_err());
    assert!(validate_whole_inventory(&[(BackboneFull { layer: 0 }, 257), expected[1]], &expected, 769).is_err());
    let overflow = [(BackboneFull { layer: 0 }, usize::MAX), (BackboneFull { layer: 1 }, 1)];
    assert!(validate_whole_inventory(&overflow, &overflow, usize::MAX).is_err());
    validate_whole_inventory(&[], &[], 0)?;
    Ok(())
}

#[test]
fn coordinator_host_admission_charges_every_concurrent_receive_ring() -> Result<()> {
    for gpus in [1, 2] {
        let plans = (0..gpus).map(|_| None).collect::<Vec<_>>();
        let peak = gpus * 128 * (1 << 20);
        assert_eq!(CoordinatorExpertPlan::admit_host(&plans.iter().map(Option::as_ref).collect::<Vec<_>>(), &[], 128, 1 << 20, peak as u64)?, peak);
        assert!(CoordinatorExpertPlan::admit_host(&plans.iter().map(Option::as_ref).collect::<Vec<_>>(), &[], 128, 1 << 20, peak as u64 - 1).is_err());
        assert!(CoordinatorExpertPlan::admit_host(&plans.iter().map(Option::as_ref).collect::<Vec<_>>(), &[], usize::MAX, 2, u64::MAX).is_err());
    }
    Ok(())
}

#[test]
fn dense_host_geometry_tracks_pools_and_large_quant_tables_without_allocating() -> Result<()> {
    use crate::shared::{experts::rtx::dense::{HostStaging, Job, RankLayout, Rect, Step, STEP_BYTES},
        fp8_load_quant::{Layout, Target}, fp8_linear::Fp8Scales};
    let buffer = cuteafd_ffi::CuteafdDeviceBuffer { device_id: 0, ..Default::default() };
    let target = Target { q: buffer, q_stride: 128, scales: buffer, kmajor: None };
    let rect = Rect { name: "rows".into(), path: "header-only".into(), base: 0,
        row_bytes: 256, rows: 0..131072, cols: 0..256 };
    let step = Step { device: 0, rects: vec![rect], jobs: vec![Job::Quant { rows: 0..131072,
        pitch: 128, col0: 0, cols: 128, layout: Layout::Channel, rule: Fp8Scales::Amax, target }] };
    let empty = RankLayout::new(vec![], vec![], 0)?;
    assert_eq!(empty.host_staging, HostStaging::default());
    for count in [1, 2, 8, 11] {
        let layout = RankLayout::new(vec![], vec![step.clone(); count], 0)?;
        let staging = layout.plan().host_staging;
        assert_eq!(staging.staging_bytes, count * STEP_BYTES);
        assert_eq!(staging.largest_step, STEP_BYTES);
        assert!(staging.table_bytes > 1 << 20, "row tables are not capped at the allocation floor");
        assert_eq!(staging.table_bytes, crate::shared::fp8_load_quant::quant_table_bytes(131072, Layout::Channel)?);
        assert_eq!(staging.bytes()?, count * STEP_BYTES + staging.table_bytes);
    }
    let block = crate::shared::fp8_load_quant::quant_table_bytes(131072, Layout::Block)?;
    assert_eq!(block, 1 << 20);
    assert!(crate::shared::fp8_load_quant::quant_table_bytes(usize::MAX, Layout::Channel).is_err());
    assert!(crate::shared::fp8_load_quant::scale_code_table_bytes(usize::MAX).is_err());
    assert!(HostStaging { staging_bytes: usize::MAX, largest_step: 2, table_bytes: 1 }.bytes().is_err());
    Ok(())
}

#[test]
fn staging_uses_actual_sizes_and_charges_disconnected_transforms_in_full() -> Result<()> {
    use crate::shared::experts::rtx::dense::{HostStaging, Rect};
    let rect = |base, bytes| Rect { name: "scale".into(), path: "shard".into(), base,
        row_bytes: bytes, rows: 0..1, cols: 0..bytes };
    let steps = (0..4096).map(|i| vec![rect(i * 122880, 122880)]).collect::<Vec<_>>();
    let bound = HostStaging::for_steps(steps.iter().map(Vec::as_slice), 1 << 20)?;
    assert_eq!(bound.largest_step, 122880);
    assert_eq!(bound.staging_bytes, 128 * ((1 << 20) + 2 * 122880) + 122880);
    assert!(bound.bytes()? < 160 << 20);
    let mixed = (0..1024).map(|i| vec![rect(i * 4096, 4096), rect((1 << 30) + i * 4096, 4096)])
        .collect::<Vec<_>>();
    assert_eq!(HostStaging::for_steps(mixed.iter().map(Vec::as_slice), 0)?.staging_bytes, 1024 * 8192);
    let overlapping = vec![vec![rect(0, 4096)]; 1024];
    assert_eq!(HostStaging::for_steps(overlapping.iter().map(Vec::as_slice), 0)?.staging_bytes, 1024 * 4096);
    Ok(())
}

#[test]
fn coordinator_host_admission_adds_dense_pools_and_exl3_validation_for_every_rank() -> Result<()> {
    use crate::shared::experts::rtx::dense::{HostStaging, STEP_BYTES};
    let root = tempfile::tempdir()?;
    let (tensors, _) = qwen4_exl3(1, 4);
    let tensors = tensors.into_iter().filter(|t| !t.0.contains(".experts.") || t.0.contains(".experts.0."))
        .collect::<Vec<_>>();
    let mut config = qwen4_config(1);
    config["text_config"]["num_experts"] = serde_json::json!(1);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    write_snapshot(root.path(), &config, &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    for gpus in [1, 2] {
        let plans = (0..gpus).map(|gpu| if gpus == 1 {
            CoordinatorExpertPlan::whole(vec![catalog.exl3().unwrap().residency(V41Exl3Layer::Backbone(0), 1, 0)?], usize::MAX)
        } else { CoordinatorExpertPlan::new(&catalog, &[0], gpu, usize::MAX) }).collect::<Result<Vec<_>>>()?;
        let owners = plans.iter().map(Some).collect::<Vec<_>>();
        let dense = vec![HostStaging { staging_bytes: 12 * STEP_BYTES, largest_step: STEP_BYTES, table_bytes: 2 << 20 }; gpus];
        let validation = plans.iter().map(|p| p.validation.bytes()).sum::<usize>();
        assert!(validation > 0);
        let peak = gpus * (128 * (1 << 20) + 12 * STEP_BYTES + (2 << 20)) + validation;
        assert_eq!(CoordinatorExpertPlan::admit_host(&owners, &dense, 128, 1 << 20, peak as u64)?, peak);
        assert!(CoordinatorExpertPlan::admit_host(&owners, &dense, 128, 1 << 20, peak as u64 - 1).is_err());
    }
    Ok(())
}

#[test]
fn host_admission_refusal_declines_peers_before_returning_a_typed_planning_error() -> Result<()> {
    use crate::shared::experts::rtx::dense::HostStaging;
    let (first_driver, mut first) = control_pair()?;
    let (second_driver, mut second) = control_pair()?;
    let driver = std::thread::spawn(move || cuteafd_transport::worker_selection::push::coordinate(
        &mut [first_driver, second_driver], |_| anyhow::bail!("refused admission must not create a group")));
    let peer = std::thread::spawn(move || -> Result<()> {
        let offer = cuteafd_transport::worker_selection::push::Offer { rank: 1, host: "spark".into(),
            arena_bytes: 1, placement: vec![] };
        assert!(planned_offer(&mut second, Ok(()), Ok(offer))?.is_none());
        Ok(())
    });
    let planned = CoordinatorExpertPlan::admit_host(&[None], &[HostStaging { staging_bytes: 8 << 20, largest_step: 1 << 20, table_bytes: 1 << 20 }],
        128, 1 << 20, 128 << 20);
    let offer = planned.as_ref().map(|_| unreachable!("staging must exceed this ring-only budget"))
        .map_err(|error| format!("{error:#}"));
    let error = planned_offer(&mut first, planned, offer).unwrap_err();
    assert!(matches!(error.downcast_ref::<PushStartupError>(), Some(PushStartupError::Planning(_))));
    peer.join().map_err(|_| anyhow::anyhow!("peer panicked"))??;
    driver.join().map_err(|_| anyhow::anyhow!("driver panicked"))??;
    Ok(())
}

fn control_pair() -> Result<(TcpStream, TcpStream)> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let client = TcpStream::connect(listener.local_addr()?)?;
    let (worker, _) = listener.accept()?;
    for stream in [&client, &worker] {
        stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    }
    Ok((client, worker))
}

fn read_control(stream: &TcpStream) -> Result<serde_json::Value> {
    use std::io::BufRead;
    let mut line = Vec::new();
    std::io::BufReader::new(stream).read_until(b'\n', &mut line)?;
    Ok(serde_json::from_slice(&line)?)
}

fn write_control(stream: &mut TcpStream, value: &serde_json::Value) -> Result<()> {
    use std::io::Write;
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    stream.write_all(&bytes)?;
    Ok(())
}

/// Every run of `plan` for `rank`, packed into slots in plan order (a single
/// source streaming in file order).
fn pack(plan: &CheckedPlan, root: &std::path::Path, rank: u32, slot_bytes: usize) -> Result<Vec<Vec<u8>>> {
    let sources = plan.plan.files.iter().map(|name| std::fs::read(root.join(name)))
        .collect::<std::io::Result<Vec<_>>>()?;
    let mut packed = Vec::new();
    let mut packer = slot::Packer::new(vec![0; slot_bytes]);
    for entry in plan.plan.entries.iter().filter(|e| e.rank == rank) {
        let pattern = &plan.plan.patterns[entry.pattern as usize];
        for (src, dst) in pattern.runs() {
            let mut bytes = &sources[entry.file as usize][(entry.src_base + src) as usize
                ..(entry.src_base + src + u64::from(pattern.run)) as usize];
            let mut destination = entry.dst_base + dst;
            while !bytes.is_empty() {
                let n = packer.push(destination, bytes);
                if n == 0 { packed.push(packer.finish().0); packer = slot::Packer::new(vec![0; slot_bytes]); continue; }
                destination += n as u64;
                bytes = &bytes[n..];
            }
        }
    }
    if !packer.is_empty() { packed.push(packer.finish().0); }
    Ok(packed)
}

/// Two coordinator GPUs (group ranks 6 and 7 after six Spark ranks) receive
/// concurrently from one fake SparkNest, each into its own fake device. Both
/// publish mount-exact EXL3 TP2 halves on success; on a failed commit both drop
/// their arenas before the drained acknowledgment.
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn coordinator_tp2_pair_is_mount_exact_on_two_devices_and_drops_before_retry() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    for device in 0..2 { fixture.configure_memory(&library, device, 64 << 30)?; }
    std::fs::write(fixture.directory().join("events"), b"")?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (tensors, _) = qwen4_exl3(2, 4);
    let tensors: Vec<_> = tensors.into_iter().filter(|tensor| !tensor.0.contains(".experts.")
        || tensor.0.contains(".experts.0.")).collect();
    let mut config = qwen4_config(2);
    config["text_config"]["num_experts"] = serde_json::json!(1);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    write_snapshot(root.path(), &config, &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    for tensor in catalog.tensors().iter().filter(|t| t.metadata.name.contains(".experts.")) {
        let source = Builder::v41_source(&catalog, &tensor.metadata.name)?;
        let image = if tensor.metadata.name.ends_with(".mcg") { 0xcbac1fedu32.to_le_bytes().to_vec() }
            else if tensor.metadata.name.ends_with(".trellis") {
                (0..source.bytes).map(|i| (i.wrapping_mul(29) >> 2) as u8 ^ 0x5a).collect()
            } else { [0u8, 0x3c].into_iter().cycle().take(source.bytes as usize).collect() };
        File::options().write(true).open(source.path)?.write_all_at(&image, source.offset)?;
    }
    let (sparks, world) = (6usize, 8usize);
    for fail in [false, true] {
        let plans = (0..2).map(|gpu| CoordinatorExpertPlan::new(&catalog, &[0, 1], gpu, usize::MAX)).collect::<Result<Vec<_>>>()?;
        let checked = plans.iter().enumerate().map(|(gpu, plan)| plan.checked(&catalog, root.path(), world, sparks + gpu))
            .collect::<Result<Vec<_>>>()?;
        // Independent mount reference: the coordinator's legacy residency reads.
        let expected = plans.iter().map(|plan| plan.residencies.iter().map(|residency| -> Result<Vec<Vec<u8>>> {
            let mut image: Vec<Vec<u8>> = residency.buffers.iter().map(|b| vec![0; b.bytes.max(16)]).collect();
            let mut scratch = vec![0; residency.scratch_bytes(&catalog)?];
            for (index, load) in residency.loads.iter().enumerate() {
                let mut bytes = vec![0; load.bytes];
                residency.read_into(&catalog, index, &mut bytes, &mut scratch)?;
                for &(buffer, offset) in &load.destinations { image[buffer][offset..offset + load.bytes].copy_from_slice(&bytes); }
            }
            for (buffer, spec) in image.iter_mut().zip(&residency.buffers) {
                for (dst, word) in buffer.chunks_exact_mut(4).zip(&spec.initial_words) { dst.copy_from_slice(&word.to_le_bytes()); }
            }
            Ok(image)
        }).collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
        let mut servers = Vec::new();
        let mut workers = Vec::new();
        for gpu in 0..2 {
            let rank = sparks + gpu;
            let socket = root.path().join(format!("{}-{gpu}.sock", if fail { "fail" } else { "good" }))
                .to_string_lossy().into_owned();
            let listener = Listener::bind(&socket)?;
            let packed = pack(&checked[gpu], root.path(), rank as u32, DEFAULT_SLOT_BYTES)?;
            let (mut coordinator, worker) = control_pair()?;
            let server_socket = socket.clone();
            workers.push((worker, socket));
            servers.push(std::thread::spawn(move || -> Result<()> {
                let conn = crate::shared::memory::push_testing::accept_within(&listener, &server_socket, Duration::from_secs(120))?;
                conn.set_timeout(Some(Duration::from_secs(10)))?;
                let (request, fd) = conn.next_request()?.context("rank did not join")?;
                let Request::Join { slots, slot_bytes, rank: joined, .. } = request else { anyhow::bail!("expected rank Join") };
                assert_eq!(joined as usize, rank);
                let file = File::from(fd.context("rank ring fd absent")?);
                conn.send_event(&Event::Joined).context("send Joined")?;
                assert_eq!(read_control(&coordinator).context("read joined")?["message"], "joined");
                for (index, bytes) in packed.into_iter().enumerate() {
                    let index = index % slots as usize;
                    file.write_all_at(&bytes, index as u64 * u64::from(slot_bytes))?;
                    conn.send_event(&Event::Filled { slot: index as u32, len: bytes.len() as u32 }).context("send Filled")?;
                    let (release, _) = conn.next_request().context("read Release")?.context("missing slot release")?;
                    assert!(matches!(release, Request::Release { slot } if slot == index as u32));
                }
                conn.send_event(&Event::Done).context("send Done")?;
                assert_eq!(read_control(&coordinator).context("read completed")?["success"], true);
                write_control(&mut coordinator, &serde_json::json!({"message":"commit", "success":!fail})).context("write commit")?;
                if fail {
                    assert_eq!(read_control(&coordinator)?["message"], "drained");
                    write_control(&mut coordinator, &serde_json::json!({"message":"mount"}))?;
                }
                Ok(())
            }));
        }
        let before = fixture.events()?;
        // The fake DSO models CUDA queues process-wide, so the two ranks receive
        // one after the other here (each against its own concurrent fake source).
        let results = workers.into_iter().zip(plans).zip(&checked).enumerate().map(|(gpu, (((mut worker, socket), plan), checked))| {
            let client = PushClient::at(socket, Duration::from_secs(10))?;
            let device = Device { library: &library, id: gpu as i32 };
            receive_tp2(device, &catalog, plan, checked, sparks + gpu, &client, &mut worker, (7, 11), 8, DEFAULT_SLOT_BYTES)
        }).collect::<Vec<Result<_>>>();
        let served = servers.into_iter().map(|s| s.join().map_err(|_| anyhow::anyhow!("fake SparkNest panicked")).and_then(|r| r))
            .collect::<Vec<_>>();
        for (gpu, (served, received)) in served.iter().zip(&results).enumerate() {
            if let Err(e) = served { eprintln!("server {gpu}: {e:#}"); }
            if let Err(e) = received { eprintln!("rank {gpu}: {e:#}"); }
        }
        for served in served { served?; }
        let results = results.into_iter().collect::<Result<Vec<_>>>()?;
        if fail {
            assert!(results.iter().all(Option::is_none));
            assert_eq!(fixture.events()?[before.len()..].matches('d').count(), 2, "both failed arenas freed before retry");
            continue;
        }
        for (gpu, (weights, image)) in results.into_iter().zip(expected).enumerate() {
            let weights = weights.context("successful push fell back")?.exl3()?;
            assert_eq!(weights.len(), 2);
            for (weight, image) in weights.iter().zip(image) {
                for (spec, expected) in weight.layout.buffers.iter().zip(image) {
                    let mut actual = vec![0; expected.len()];
                    Device { library: &library, id: gpu as i32 }.run(|| library.copy_d2h(&mut actual, weight.buffer(&spec.name)?))?;
                    assert_eq!(actual, expected, "GPU {gpu} {}", spec.name);
                }
            }
            drop(weights);
        }
        assert_eq!(fixture.events()?[before.len()..].matches('d').count(), 2, "each GPU's rank arena freed once");
    }
    Ok(())
}

/// Dense operands recorded in host memory (a stand-in for a family's operand
/// allocations): every planned byte must arrive, at its offset, before Done.
struct HostDense {
    image: Vec<u8>,
    planned: Vec<(u64, u64)>,
    delivered: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
    device: i32,
}

impl crate::shared::memory::push_consumer::PushDestination for HostDense {
    type Delivered = ();
    fn belongs_to(&self, device: Device<'_>) -> bool { device.id == self.device }
    fn logical_bytes(&self) -> usize { self.image.len() }
    fn device_bytes(&self) -> usize { self.image.len() }
    fn check_fragment(&self, offset: u64, bytes: u64) -> Result<()> {
        anyhow::ensure!(self.planned.iter().any(|&(s, n)| offset >= s && offset + bytes <= s + n), "unplanned dense fragment");
        Ok(())
    }
    fn destination(&self, offset: usize, bytes: usize) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
        // The fake DSO copies host-to-"device" with memcpy, so a host pointer works here.
        Ok(cuteafd_ffi::CuteafdDeviceBuffer { ptr: self.image[offset..offset + bytes].as_ptr() as *mut _, bytes,
            device_id: self.device, flags: 0 })
    }
    fn copy_validation(&mut self, _: u64, _: &[u8]) -> Result<()> { anyhow::bail!("dense has no host tail") }
    fn copied(&mut self, _: u64, _: u64) -> Result<()> { Ok(()) }
    fn done(self) -> Result<()> {
        *self.delivered.lock().unwrap() = self.image.clone();
        Ok(())
    }
}

/// One GPU rank carrying dense operands in `[0, base)` and the TP2 halves from
/// `base`: both land exactly, through one ring, in one group.
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn combined_rank_lands_dense_operands_and_tp2_halves_exactly() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    fixture.configure_memory(&library, 0, 64 << 30)?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (tensors, _) = qwen4_exl3(2, 4);
    let tensors: Vec<_> = tensors.into_iter().filter(|tensor| !tensor.0.contains(".experts.")
        || tensor.0.contains(".experts.0.")).collect();
    let mut config = qwen4_config(2);
    config["text_config"]["num_experts"] = serde_json::json!(1);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    write_snapshot(root.path(), &config, &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    for tensor in catalog.tensors() {
        let image: Vec<u8> = if tensor.metadata.name.ends_with(".mcg") { 0xcbac1fedu32.to_le_bytes().to_vec() }
            else if tensor.metadata.name.ends_with(".suh") || tensor.metadata.name.ends_with(".svh") {
                [0u8, 0x3c].into_iter().cycle().take(tensor.metadata.byte_length as usize).collect()
            } else { (0..tensor.metadata.byte_length).map(|i| (i as u8).wrapping_mul(13) ^ 0x5a).collect() };
        File::options().write(true).open(root.path().join(&tensor.shard))?.write_all_at(&image, tensor.metadata.byte_offset)?;
    }
    // Dense part: two non-expert tensors landed whole, a gap between them.
    let dense_names: Vec<String> = catalog.tensors().iter().filter(|t| !t.metadata.name.contains(".experts."))
        .take(2).map(|t| t.metadata.name.clone()).collect();
    let mut planned = Vec::new();
    let mut cursor = 0u64;
    for name in &dense_names {
        let start = cursor.next_multiple_of(256);
        let bytes = catalog.tensor(name)?.metadata.byte_length;
        planned.push((start, bytes));
        cursor = start + bytes + 1;
    }
    let base = cursor.next_multiple_of(256);
    let (rank, world) = (6usize, 8usize);
    let plan = CoordinatorExpertPlan::new(&catalog, &[0, 1], 0, usize::MAX)?;
    let checked_tp2 = plan.checked(&catalog, root.path(), world, rank)?;
    let mut arenas = vec![0; world];
    arenas[rank] = base + plan.layout.logical_bytes() as u64;
    let mut builder = Builder::new(root.path(), arenas);
    for (name, &(start, bytes)) in dense_names.iter().zip(&planned) {
        builder.contiguous(&Builder::v41_source(&catalog, name)?, 0, bytes,
            cuteafd_loader::push_load::builder::Destination { rank: rank as u32, offset: start, bytes })?;
    }
    plan.append_at(&mut builder, &catalog, rank as u32, base)?;
    let combined = builder.finish()?;
    let socket = root.path().join("combined.sock").to_string_lossy().into_owned();
    let listener = Listener::bind(&socket)?;
    let packed = pack(&combined, root.path(), rank as u32, DEFAULT_SLOT_BYTES)?;
    let (mut coordinator, mut worker) = control_pair()?;
    let server_socket = socket.clone();
    let server = std::thread::spawn(move || -> Result<()> {
        let conn = crate::shared::memory::push_testing::accept_within(&listener, &server_socket, Duration::from_secs(120))?;
        conn.set_timeout(Some(Duration::from_secs(10)))?;
        let (_, fd) = conn.next_request()?.context("rank did not join")?;
        let file = File::from(fd.context("rank ring fd absent")?);
        conn.send_event(&Event::Joined)?;
        assert_eq!(read_control(&coordinator)?["message"], "joined");
        for (index, bytes) in packed.into_iter().enumerate() {
            let slot = index % 8;
            file.write_all_at(&bytes, slot as u64 * DEFAULT_SLOT_BYTES as u64)?;
            conn.send_event(&Event::Filled { slot: slot as u32, len: bytes.len() as u32 })?;
            let (release, _) = conn.next_request()?.context("missing slot release")?;
            assert!(matches!(release, Request::Release { slot: s } if s == slot as u32));
        }
        conn.send_event(&Event::Done)?;
        assert_eq!(read_control(&coordinator)?["success"], true);
        write_control(&mut coordinator, &serde_json::json!({"message":"commit", "success": true}))?;
        Ok(())
    });
    let delivered = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let dense = HostDense { image: vec![0; cursor as usize], planned: planned.clone(), delivered: delivered.clone(), device: 0 };
    let client = PushClient::at(socket, Duration::from_secs(10))?;
    let device = Device { library: &library, id: 0 };
    let weights = receive_rank(device, Some((&catalog, plan, &checked_tp2)), Some(DenseRank { destination: dense, base }),
        rank, &client, &mut worker, (7, 11), 8, DEFAULT_SLOT_BYTES)?;
    server.join().map_err(|_| anyhow::anyhow!("fake SparkNest panicked"))??;
    let weights = weights.context("combined push fell back")?.exl3()?;
    assert_eq!(weights.len(), 2);
    let image = delivered.lock().unwrap().clone();
    for (name, &(start, bytes)) in dense_names.iter().zip(&planned) {
        let tensor = catalog.tensor(name)?;
        let mut expected = vec![0; bytes as usize];
        File::open(root.path().join(&tensor.shard))?.read_exact_at(&mut expected, tensor.metadata.byte_offset)?;
        assert_eq!(&image[start as usize..(start + bytes) as usize], &expected[..], "dense {name}");
    }
    for weight in &weights {
        let mut scratch = vec![0; weight.layout.scratch_bytes(&catalog)?];
        for (index, load) in weight.layout.loads.iter().enumerate() {
            let mut expected = vec![0; load.bytes];
            weight.layout.read_into(&catalog, index, &mut expected, &mut scratch)?;
            for &(buffer, offset) in &load.destinations {
                let mut actual = vec![0; load.bytes];
                let spec = &weight.layout.buffers[buffer];
                let buffer = weight.buffer(&spec.name)?;
                // SAFETY: the load's destination range lies inside this published buffer.
                let view = cuteafd_ffi::CuteafdDeviceBuffer { bytes: load.bytes,
                    ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() }, ..buffer };
                library.copy_d2h(&mut actual, view)?;
                assert_eq!(actual, expected, "TP2 {}", load.tensor);
            }
        }
    }
    Ok(())
}

/// A dense planning failure on the coordinator must not abort the load: both
/// GPU ranks offer mount, so the group driver opens no group and sends every
/// rank (a Spark included) the mount decision, and the coordinator returns
/// `Ok(None)` (TP2 halves and dense operands then load through the mount).
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn dense_planning_error_sends_the_whole_group_to_mount() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (tensors, _) = qwen4_exl3(2, 4);
    let tensors: Vec<_> = tensors.into_iter().filter(|tensor| !tensor.0.contains(".experts.")
        || tensor.0.contains(".experts.0.")).collect();
    let mut config = qwen4_config(2);
    config["text_config"]["num_experts"] = serde_json::json!(1);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    write_snapshot(root.path(), &config, &tensors, None);
    // A SparkNest namespace (ADR-064 root identity, shards with hosts), so the
    // TP2 plans alone would be offered: only the dense error can decline them.
    let mark = |path: &std::path::Path, name: &std::ffi::CStr, value: &[u8]| -> Result<()> {
        let path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())?;
        // SAFETY: both strings are NUL-terminated and live across the call; the
        // value pointer and length describe one live buffer.
        let status = unsafe { libc::setxattr(path.as_ptr(), name.as_ptr(), value.as_ptr().cast(), value.len(), 0) };
        anyhow::ensure!(status == 0, "setxattr: {}", std::io::Error::last_os_error());
        Ok(())
    };
    mark(root.path(), c"user.sparknest.cluster", br#"{"me":"coordinator"}"#)?;
    for entry in std::fs::read_dir(root.path())? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "safetensors") { mark(&path, c"user.sparknest.hosts", b"coordinator")?; }
    }
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    dense_error_mount_group(&library, root.path(), &catalog, true)?;
    // Non-EXL3 groups carry dense-only coordinator ranks, not EXL3 TP2 halves.
    let root = tempfile::tempdir_in(fixture.directory())?;
    let catalog = crate::shared::memory::push_fp8::tests::fixture(root.path(),
        cuteafd_loader::formats::fp8_experts::ExpertFormat::Fp8Block128, false)?;
    mark(root.path(), c"user.sparknest.cluster", br#"{"me":"coordinator"}"#)?;
    dense_error_mount_group(&library, root.path(), &catalog, false)?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    cuteafd_loader::plan::testing::exlr8::fixture(root.path(), false);
    let catalog = crate::shared::memory::push_exlr8::tests::fixture_catalog(root.path())?;
    mark(root.path(), c"user.sparknest.cluster", br#"{"me":"coordinator"}"#)?;
    dense_error_mount_group(&library, root.path(), &catalog, false)?;
    Ok(())
}

fn dense_error_mount_group(library: &cuteafd_ffi::NativeLibrary, root: &std::path::Path,
    catalog: &OfficialV41Catalog, tp2: bool) -> Result<()> {
    let sparks = 1;
    if tp2 { for gpu in 0..2 {
        let plan = CoordinatorExpertPlan::new(catalog, &[0, 1], gpu, usize::MAX)?;
        let checked = plan.checked(catalog, root, sparks + 2, sparks + gpu)?;
        assert!(checked.plan.files.iter().all(|f| cuteafd_loader::push_load::is_push_shard(&root.join(f))),
            "without the dense error the TP2 halves would be offered");
    } }
    let (push, coordinator_streams, _) = CoordinatorPush::prepare(root, Some(catalog), 2,
        tp2.then(|| CoordinatorExperts::Tp2 { layers: vec![0, 1], budgets: vec![usize::MAX; 2] }))?;
    // One Spark rank offering push, then the two coordinator ranks, in group order.
    let (mut spark_driver, mut spark) = control_pair()?;
    let spark_thread = std::thread::spawn(move || cuteafd_transport::worker_selection::push::offer(&mut spark,
        Some(cuteafd_transport::worker_selection::push::Offer { rank: 0, host: "spark".into(), arena_bytes: 1,
            placement: Vec::new() })));
    let driver = std::thread::spawn(move || -> Result<bool> {
        let mut streams = vec![spark_driver.try_clone()?];
        streams.extend(coordinator_streams);
        let mut prepared = false;
        cuteafd_transport::worker_selection::push::coordinate(&mut streams, |_| {
            prepared = true;
            anyhow::bail!("no group may open after a declined coordinator rank")
        })?;
        drop(spark_driver.shutdown(std::net::Shutdown::Both));
        Ok(prepared)
    });
    let devices = [Device { library, id: 0 }, Device { library, id: 1 }];
    let dense: Option<Result<(Vec<DensePlan>, Vec<NoDense>)>> = Some(Err(anyhow::anyhow!("injected dense planning error")));
    let (pushed, ()) = push.receive_with(devices.to_vec(), Some(catalog), sparks, dense, || ());
    assert!(pushed.is_err(), "an uncommitted required coordinator group must fail startup after draining");
    assert!(!driver.join().map_err(|_| anyhow::anyhow!("driver panicked"))??, "the group driver must not prepare a group");
    assert_eq!(spark_thread.join().map_err(|_| anyhow::anyhow!("spark panicked"))??, None,
        "the Spark rank receives the whole-group mount decision");
    Ok(())
}

#[test]
fn pre_membership_dense_planning_error_aborts_required_startup() {
    let planned = PushPolicy::Required.plan::<(CoordinatorPush, Vec<TcpStream>,
        crate::shared::memory::push_exl3::CoordinatorRanks)>(|| {
        anyhow::bail!("injected dense planning error")
    });
    assert!(matches!(planned, Err(PushStartupError::Planning(_))),
        "failed required planning must not become a Spark-only group plus mount loading");
}

/// A model of SparkNest's push credit flow (crates/nest-data/src/push.rs
/// `grant`/`flush`: today's policy as of 0.6.2, and the per-(source, rank)
/// floor of 1d3be3b), NOT a cuteafd guarantee: it documents why coordinator
/// ranks stalled the K4 and Flash push groups and checks the floor's fix.
///
/// - Each rank host has one incast budget (in slots) shared by its ranks; a
///   fill frees one unit; `grant(rank)` runs at join and on that rank's own
///   releases only, handing free ring slots round robin to EVERY source owing
///   the rank (needed yet or not) while the budget allows (at least one grant
///   if the host has none out).
/// - With the floor, `grant(rank)` first tops each owing source up to one held
///   grant of that rank, outside the budget, then runs the budget's round robin.
/// - Each source streams its slots in plan order and blocks in `flush` until it
///   holds a credit for that slot's rank. The engine releases each fill at once.
mod sparknest_credit_model {
    use std::collections::VecDeque;

    #[derive(Clone, Copy, PartialEq)]
    pub enum Policy {
        /// SparkNest 0.6.2: one budget per host, no floor.
        Budget,
        /// SparkNest 1d3be3b: each owing source held at one grant outside the budget.
        SourceFloor,
    }

    /// Returns true when every slot was delivered, false on a stall (no source
    /// can make progress and no grant can be issued).
    pub fn run(policy: Policy, host_of: &[usize], ring: usize, budget: usize, streams: &[Vec<usize>]) -> bool {
        let ranks = host_of.len();
        let hosts = host_of.iter().max().map_or(0, |h| h + 1);
        let sources = streams.len();
        let mut free = vec![ring; ranks];
        let mut out = vec![0usize; hosts];
        let mut next = vec![0usize; ranks];
        // credits[source][rank] = (budgeted, floor) grants held.
        let mut credits = vec![vec![(0usize, 0usize); ranks]; sources];
        let mut cursor = vec![0usize; sources];
        let owing: Vec<Vec<usize>> = (0..ranks).map(|r| (0..sources)
            .filter(|&s| streams[s].contains(&r)).collect()).collect();
        let grant = |r: usize, free: &mut Vec<usize>, out: &mut Vec<usize>, next: &mut Vec<usize>,
            credits: &mut Vec<Vec<(usize, usize)>>| {
            if owing[r].is_empty() { return; }
            if policy == Policy::SourceFloor {
                for &s in &owing[r] {
                    let held = credits[s][r].0 + credits[s][r].1;
                    if held == 0 && free[r] > 0 { free[r] -= 1; credits[s][r].1 += 1; }
                }
            }
            let h = host_of[r];
            while free[r] > 0 && (out[h] < budget || (out[h] == 0 && policy == Policy::Budget)) {
                let s = owing[r][next[r] % owing[r].len()];
                next[r] += 1;
                free[r] -= 1;
                out[h] += 1;
                credits[s][r].0 += 1;
            }
        };
        for r in 0..ranks { grant(r, &mut free, &mut out, &mut next, &mut credits); }
        let mut released: VecDeque<usize> = VecDeque::new();
        loop {
            let mut progressed = false;
            for s in 0..sources {
                while let Some(&r) = streams[s].get(cursor[s]) {
                    let held = &mut credits[s][r];
                    if held.1 > 0 { held.1 -= 1; }
                    else if held.0 > 0 { held.0 -= 1; out[host_of[r]] -= 1; } // a budgeted fill frees budget
                    else { break; }
                    cursor[s] += 1;
                    released.push_back(r);
                    progressed = true;
                }
            }
            while let Some(r) = released.pop_front() {
                free[r] += 1;
                grant(r, &mut free, &mut out, &mut next, &mut credits);
            }
            if cursor.iter().zip(streams).all(|(c, s)| *c == s.len()) { return true; }
            if !progressed { return false; }
        }
    }

    /// Plan-order streams: each source's files interleave both ranks' slots
    /// (a dense tensor's row shares for GPU0 then GPU1, file after file).
    pub fn interleaved(sources: usize, slots: usize, ranks: &[usize], ahead: usize) -> Vec<Vec<usize>> {
        (0..sources).map(|s| {
            let mut v = Vec::new();
            for block in 0..slots / ahead {
                for &r in ranks { v.extend(std::iter::repeat_n(r, ahead + (block + s) % 3)); }
            }
            v
        }).collect()
    }
}

#[test]
fn sparknest_model_two_ranks_on_one_host_stall_under_the_shared_budget() {
    use sparknest_credit_model::{interleaved, run, Policy};
    // Group ...667's shape: six sources streaming slots for two coordinator
    // ranks on one host (ring 128 slots, incast budget 64 slots).
    let streams = interleaved(6, 400, &[0, 1], 20);
    assert!(!run(Policy::Budget, &[0, 0], 128, 64, &streams),
        "one shared budget is expected to stall two rings on one host");
    assert!(run(Policy::SourceFloor, &[0, 0], 128, 64, &streams),
        "the per-(source, rank) floor outside the budget must complete");
}

/// The 1-RTX GLM Flash pair's shape (group ...669): one coordinator rank on its
/// own host beside two Spark ranks. One source (kiwi there) is in a long
/// stretch of coordinator tensors while the other five stream experts. Grants
/// go round robin to every source owing a rank, needed yet or not, so each
/// Spark rank's budget parks at the coordinator-busy source and the
/// coordinator's at the expert-busy ones: a stall under today's policy (as
/// SparkNest's grants_parked_at_busy_sources_do_not_deadlock_one_rank_per_host
/// reproduces at 1d3be3b); the per-(source, rank) floor (1d3be3b) completes.
#[test]
fn sparknest_model_one_coordinator_rank_beside_spark_ranks() {
    use sparknest_credit_model::{run, Policy};
    // Ranks 0 and 1 are Sparks, rank 2 the coordinator; source 3 holds a long
    // run of coordinator tensors before its expert files, the rest the reverse.
    let streams: Vec<Vec<usize>> = (0..6).map(|s| {
        let experts: Vec<usize> = (0..400).map(|i| i % 2).collect();
        let dense = vec![2; 300];
        if s == 3 { [dense, experts].concat() } else { [experts, dense].concat() }
    }).collect();
    let hosts = [0, 1, 2];
    assert!(!run(Policy::Budget, &hosts, 128, 64, &streams),
        "grants parked at busy sources are expected to stall under today's policy");
    assert!(run(Policy::SourceFloor, &hosts, 128, 64, &streams),
        "a per-(source, rank) credit floor outside the shared budget must complete");
}


/// One coordinator GPU's rank (dense records and/or TP2 halves, laid out as
/// the group driver lays them out) received through a ring from a fake
/// SparkNest that streams the rank's checked plan in plan order, then
/// committed. For installed-checkpoint goldens on real or fake devices.
pub(crate) fn receive_pushed<'a>(device: Device<'a>, catalog: Option<&cuteafd_loader::OfficialV41Catalog>,
    root: &std::path::Path, socket_dir: &std::path::Path, records: &crate::shared::experts::rtx::dense::Records,
    tp2: Option<CoordinatorExpertPlan>, sparks: usize, gpu: usize) -> Result<PushedExperts<'a>> {
    use crate::shared::experts::rtx::dense::RankLayout;
    let rank = sparks + gpu;
    let layout = RankLayout::new(records.operands.clone(), records.steps.clone(), device.id)?
        .with_finishes(&records.finishes);
    let dense_plan = layout.plan();
    let (base, host) = (dense_plan.base, dense_plan.host_bytes);
    let tp2_bytes = tp2.as_ref().map_or(0, |p| p.logical_bytes());
    let (host_base, total) = rank_bytes(base, tp2_bytes, host);
    // The rank's checked plan (built twice: one copy streams from the fake source).
    let build = || -> Result<CheckedPlan> {
        let mut arenas = vec![0; sparks + 2];
        arenas[rank] = total;
        let mut builder = Builder::new(root, arenas);
        (dense_plan.append)(&mut builder, rank as u32, host_base)?;
        if let Some(plan) = &tp2 { plan.append_at(&mut builder, catalog.context("TP2 replay needs the expert catalog")?, rank as u32, base)?; }
        Ok(builder.finish()?)
    };
    let checked = build()?;
    // As plan_records: the dense destination's coverage from its own plan alone.
    let expected = crate::shared::memory::weight_arena::ExpectedCoverage::from_plan(
        &layout.checked(root, sparks + 2, rank)?, rank as u32)?;
    let dense = crate::shared::experts::rtx::dense::DenseArena::new(device, layout, expected)?;
    let socket = socket_dir.join(format!("g{gpu}.sock")).to_string_lossy().into_owned();
    let listener = Listener::bind(&socket)?;
    let (mut coordinator, mut worker) = control_pair()?;
    for stream in [&coordinator, &worker] { stream.set_read_timeout(Some(Duration::from_secs(600)))?; }
    let streamed = build()?;
    let source_root = root.to_path_buf();
    let server_socket = socket.clone();
    let server = std::thread::spawn(move || -> Result<()> {
        let conn = crate::shared::memory::push_testing::accept_within(&listener, &server_socket, Duration::from_secs(300))?;
        conn.set_timeout(Some(Duration::from_secs(600)))?;
        let (request, fd) = conn.next_request()?.context("rank did not join")?;
        let Request::Join { slots, slot_bytes, .. } = request else { anyhow::bail!("expected rank Join") };
        let file = File::from(fd.context("rank ring fd absent")?);
        conn.send_event(&Event::Joined)?;
        assert_eq!(read_control(&coordinator)?["message"], "joined");
        // Streams the rank's runs in plan order, one slot at a time (bounded memory).
        let mut index = 0usize;
        let mut send = |bytes: Vec<u8>| -> Result<()> {
            let at = index % slots as usize;
            file.write_all_at(&bytes, at as u64 * u64::from(slot_bytes))?;
            conn.send_event(&Event::Filled { slot: at as u32, len: bytes.len() as u32 })?;
            let (release, _) = conn.next_request()?.context("missing slot release")?;
            anyhow::ensure!(matches!(release, Request::Release { slot } if slot == at as u32), "release order");
            index += 1;
            Ok(())
        };
        let mut sources = std::collections::HashMap::new();
        let mut packer = slot::Packer::new(vec![0; slot_bytes as usize]);
        // SparkNest streams each file's runs in source-offset order (one source
        // per file): replay that order, not the plan's entry order.
        let mut runs = Vec::new();
        for entry in streamed.plan.entries.iter().filter(|e| e.rank == rank as u32) {
            let pattern = &streamed.plan.patterns[entry.pattern as usize];
            for (src, dst) in pattern.runs() { runs.push((entry.file, entry.src_base + src, entry.dst_base + dst, pattern.run)); }
        }
        runs.sort_unstable();
        for (file, src, dst, length) in runs {
            let source = sources.entry(file).or_insert_with(||
                File::open(source_root.join(&streamed.plan.files[file as usize])));
            let source = source.as_ref().map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut run = vec![0u8; length as usize];
            source.read_exact_at(&mut run, src)?;
            let (mut bytes, mut destination) = (&run[..], dst);
            while !bytes.is_empty() {
                let n = packer.push(destination, bytes);
                if n == 0 {
                    send(std::mem::replace(&mut packer, slot::Packer::new(vec![0; slot_bytes as usize])).finish().0)?;
                    continue;
                }
                destination += n as u64;
                bytes = &bytes[n..];
            }
        }
        if !packer.is_empty() { send(packer.finish().0)?; }
        conn.send_event(&Event::Done)?;
        anyhow::ensure!(read_control(&coordinator)?["success"] == true, "rank candidate rejected");
        write_control(&mut coordinator, &serde_json::json!({"message":"commit", "success": true}))?;
        Ok(())
    });
    let client = PushClient::at(socket.clone(), Duration::from_secs(600))?;
    // As the group driver: the TP2 arena's own checked entries (from offset 0).
    let tp2_checked = tp2.as_ref().map(|plan| plan.checked(catalog.context("TP2 replay needs the expert catalog")?, root, sparks + 2, rank)).transpose()?;
    let tp2 = match (tp2, tp2_checked.as_ref()) { (Some(plan), Some(checked)) => Some((catalog.context("TP2 catalog")?, plan, checked)), _ => None };
    let received = receive_rank(device, tp2, Some(DenseRank { destination: dense, base }), rank, &client,
        &mut worker, (7, 11), 8, DEFAULT_SLOT_BYTES);
    if !server.is_finished() {
        // A rank that failed before joining never connects: wake the fake source.
        let _ = cuteafd_loader::push_load::Conn::connect(&socket);
    }
    let served = server.join().map_err(|_| anyhow::anyhow!("fake SparkNest panicked"))?;
    let received = received?;
    served?;
    received.context("pushed rank fell back")
}

/// `receive_pushed` (the installed-golden replay driver) on a synthetic
/// snapshot: dense operands, a staged step and TP2 halves in one rank.
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn receive_pushed_replays_dense_records_and_tp2_halves() -> Result<()> {
    use crate::shared::experts::rtx::dense::{Job, Operand, Rect, Records, Step};
    let _ = tracing_subscriber::fmt().with_env_filter("info").with_test_writer().try_init();
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    fixture.configure_memory(&library, 0, 64 << 30)?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (tensors, _) = qwen4_exl3(2, 4);
    let tensors: Vec<_> = tensors.into_iter().filter(|tensor| !tensor.0.contains(".experts.")
        || tensor.0.contains(".experts.0.")).collect();
    let mut config = qwen4_config(2);
    config["text_config"]["num_experts"] = serde_json::json!(1);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    write_snapshot(root.path(), &config, &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    for tensor in catalog.tensors().iter().filter(|t| t.metadata.name.ends_with(".mcg")) {
        let source = Builder::v41_source(&catalog, &tensor.metadata.name)?;
        File::options().write(true).open(source.path)?.write_all_at(&0xcbac1fedu32.to_le_bytes(), source.offset)?;
    }
    let device = Device { library: &library, id: 0 };
    let dense = catalog.tensors().iter().filter(|t| !t.metadata.name.contains(".experts.")).take(3)
        .map(|t| Rect { name: t.metadata.name.clone(), path: root.path().join(&t.shard), base: t.metadata.byte_offset,
            row_bytes: t.metadata.byte_length as usize, rows: 0..1, cols: 0..t.metadata.byte_length as usize })
        .collect::<Vec<_>>();
    let allocations = dense.iter().map(|r| crate::shared::memory::DeviceAllocation::new(&library, r.bytes().max(256)))
        .collect::<Result<Vec<_>>>()?;
    let mut records = Records::default();
    anyhow::ensure!(dense.len() == 2, "fixture has two dense tensors");
    records.operands.push(Operand { device: 0, buffer: allocations[0].buffer, lands: vec![(dense[0].clone(), 0)],
        bytes: dense[0].bytes() });
    records.steps.push(Step { device: 0, rects: vec![dense[1].clone()],
        jobs: vec![Job::Upload { at: 0, bytes: dense[1].bytes(), dst: allocations[1].buffer }] });
    let plan = CoordinatorExpertPlan::new(&catalog, &[0, 1], 0, usize::MAX)?;
    let sockets = tempfile::tempdir_in(fixture.directory())?;
    let pushed = receive_pushed(device, Some(&catalog), root.path(), sockets.path(), &records, Some(plan), 1, 0)?;
    assert_eq!(pushed.exl3()?.len(), 2);
    for (rect, allocation) in dense.iter().zip(&allocations) {
        let mut expected = vec![0; rect.bytes()];
        rect.read(&mut expected)?;
        let mut actual = vec![0; rect.bytes()];
        library.copy_d2h(&mut actual, cuteafd_ffi::CuteafdDeviceBuffer { bytes: rect.bytes(), ..allocation.buffer })?;
        assert_eq!(actual, expected, "{}", rect.name);
    }
    Ok(())
}

/// A fake source no rank connects to fails within its bound, not forever.
#[test]
fn fake_source_accept_is_bounded() -> Result<()> {
    let root = tempfile::tempdir()?;
    let socket = root.path().join("idle.sock").to_string_lossy().into_owned();
    let listener = Listener::bind(&socket)?;
    let started = std::time::Instant::now();
    let error = crate::shared::memory::push_testing::accept_within(&listener, &socket, Duration::from_secs(1)).err().context("an idle source must fail")?;
    assert!(error.to_string().contains("no rank connected"), "{error}");
    assert!(started.elapsed() < Duration::from_secs(10));
    Ok(())
}

/// Upload-only steps require no transform staging, even when their checkpoint
/// rectangles live far apart. Both mixed and split recipes land bytes directly.
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn scales_first_shard_uploads_land_without_transform_staging() -> Result<()> {
    use crate::shared::experts::rtx::dense::{testing::PEAKS, Job, Operand, Rect, Records, Step};
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    fixture.configure_memory(&library, 0, 64 << 30)?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    // A SparkNest-shaped namespace with one shard: 16 scales (4 KiB each), then 16 weights (256 KiB each).
    let experts = 16;
    let (scale, weight) = (4096usize, 256usize << 10);
    let shard = root.path().join("model-00001-of-00001.safetensors");
    std::fs::write(&shard, (0..experts * (scale + weight)).map(|i| (i % 251) as u8).collect::<Vec<_>>())?;
    let mark = |path: &std::path::Path, name: &std::ffi::CStr, value: &[u8]| -> Result<()> {
        let path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())?;
        // SAFETY: NUL-terminated strings and one live value buffer for the call.
        let status = unsafe { libc::setxattr(path.as_ptr(), name.as_ptr(), value.as_ptr().cast(), value.len(), 0) };
        anyhow::ensure!(status == 0, "setxattr: {}", std::io::Error::last_os_error());
        Ok(())
    };
    mark(root.path(), c"user.sparknest.cluster", br#"{"me":"coordinator"}"#)?;
    mark(&shard, c"user.sparknest.hosts", b"coordinator")?;
    let rect = |base: usize, bytes: usize| Rect { name: format!("t{base}"), path: shard.clone(), base: base as u64,
        row_bytes: bytes, rows: 0..1, cols: 0..bytes };
    let (scale_at, weight_at) = (|e: usize| e * scale, |e: usize| experts * scale + e * weight);
    let catalog = cuteafd_loader::read_expert_catalog(&{
        // Any expert catalog serves only as the replay driver's argument (no TP2 here).
        let synthetic = tempfile::tempdir_in(fixture.directory())?;
        let (tensors, _) = qwen4_exl3(2, 4);
        write_snapshot(synthetic.path(), &qwen4_config(2), &tensors, None);
        synthetic.keep()
    })?;
    let device = Device { library: &library, id: 0 };
    let sockets = tempfile::tempdir_in(fixture.directory())?;
    // Old shape: each step stages [scale (front), weight (back)].
    let staged = crate::shared::memory::DeviceAllocation::new(&library, experts * (scale + weight))?;
    let mut mixed = Records::default();
    for e in 0..experts {
        // SAFETY: expert e's complete image lies inside the retained allocation.
        let dst = cuteafd_ffi::CuteafdDeviceBuffer { ptr: unsafe {
            staged.buffer.ptr.cast::<u8>().add(e * (scale + weight)).cast()
        }, bytes: scale + weight, ..staged.buffer };
        mixed.steps.push(Step { device: 0, rects: vec![rect(scale_at(e), scale), rect(weight_at(e), weight)],
            jobs: vec![Job::Upload { at: 0, bytes: scale + weight, dst }] });
    }
    receive_pushed(device, Some(&catalog), root.path(), sockets.path(), &mixed, None, 1, 0)?;
    assert_eq!(PEAKS.with(|p| p.take()), vec![(0, 0)]);
    let file = std::fs::read(&shard)?;
    let mut actual = vec![0; experts * (scale + weight)];
    library.copy_d2h(&mut actual, staged.buffer)?;
    for e in 0..experts {
        let base = e * (scale + weight);
        assert_eq!(&actual[base..base + scale], &file[scale_at(e)..scale_at(e) + scale]);
        assert_eq!(&actual[base + scale..base + scale + weight], &file[weight_at(e)..weight_at(e) + weight]);
    }
    // New shape: weights-only steps, scales as operands.
    let scales = crate::shared::memory::DeviceAllocation::new(&library, experts * scale)?;
    let weights = crate::shared::memory::DeviceAllocation::new(&library, experts * weight)?;
    let mut split = Records::default();
    for e in 0..experts {
        // SAFETY (offsets): expert e's slices lie inside both allocations.
        let at = |a: &crate::shared::memory::DeviceAllocation<'_>, off: usize, bytes: usize| cuteafd_ffi::CuteafdDeviceBuffer {
            ptr: unsafe { a.buffer.ptr.cast::<u8>().add(off) }.cast(), bytes, ..a.buffer };
        split.operands.push(Operand { device: 0, buffer: at(&scales, e * scale, scale), bytes: scale,
            lands: vec![(rect(scale_at(e), scale), 0)] });
        split.steps.push(Step { device: 0, rects: vec![rect(weight_at(e), weight)],
            jobs: vec![Job::Upload { at: 0, bytes: weight, dst: at(&weights, e * weight, weight) }] });
    }
    receive_pushed(device, Some(&catalog), root.path(), sockets.path(), &split, None, 1, 0)?;
    let peaks = PEAKS.with(|p| p.take());
    assert_eq!(peaks.len(), 1);
    let (staged, pinned) = peaks[0];
    assert_eq!((staged, pinned), (0, 0), "uploads bypass whole-step staging");
    let mut actual = vec![0u8; experts * weight];
    library.copy_d2h(&mut actual, weights.buffer)?;
    let file = std::fs::read(&shard)?;
    assert_eq!(&actual[..], &file[experts * scale..], "weights landed exactly");
    let mut actual = vec![0u8; experts * scale];
    library.copy_d2h(&mut actual, scales.buffer)?;
    assert_eq!(&actual[..], &file[..experts * scale], "scales landed exactly");
    Ok(())
}

/// Without Spark peers the coordinator GPUs form the push group alone: the
/// owner's driver offers both ranks, creates the group on the SparkNest
/// endpoint and uploads their dense entries; each rank joins, receives its
/// runs in file order and commits, and every operand lands exactly.
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn coordinator_ranks_alone_push_dense_records_exactly() -> Result<()> {
    use crate::shared::experts::rtx::dense::{plan_records, Job, Operand, Rect, Records, Step};
    let _ = tracing_subscriber::fmt().with_env_filter("info").with_test_writer().try_init();
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    for device in 0..2 { fixture.configure_memory(&library, device, 64 << 30)?; }
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (tensors, _) = qwen4_exl3(2, 4);
    let tensors: Vec<_> = tensors.into_iter().filter(|t| !t.0.contains(".experts.")
        || t.0.contains(".experts.0.")).collect();
    let mut config = qwen4_config(2);
    config["text_config"]["num_experts"] = serde_json::json!(1);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    write_snapshot(root.path(), &config, &tensors, None);
    let mark = |path: &std::path::Path, name: &std::ffi::CStr, value: &[u8]| -> Result<()> {
        let path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())?;
        // SAFETY: both strings are NUL-terminated and live across the call; the
        // value pointer and length describe one live buffer.
        let status = unsafe { libc::setxattr(path.as_ptr(), name.as_ptr(), value.as_ptr().cast(), value.len(), 0) };
        anyhow::ensure!(status == 0, "setxattr: {}", std::io::Error::last_os_error());
        Ok(())
    };
    mark(root.path(), c"user.sparknest.cluster", br#"{"me":"coordinator"}"#)?;
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(root.path())? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "safetensors") {
            mark(&path, c"user.sparknest.hosts", b"coordinator")?;
            let bytes = std::fs::metadata(&path)?.len();
            let pattern: Vec<u8> = (0..bytes).map(|i| (i as u8).wrapping_mul(29) ^ 0xa5).collect();
            // Keep the header; pattern the payload.
            let header = 8 + u64::from_le_bytes(std::fs::read(&path)?[..8].try_into()?) as usize;
            File::options().write(true).open(&path)?.write_all_at(&pattern[header..], header as u64)?;
            shards.push(path);
        }
    }
    // Four non-expert tensors, from the shards' own headers.
    let mut names = Vec::new();
    for path in &shards {
        let bytes = std::fs::read(path)?;
        let length = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&bytes[8..8 + length])?;
        for (name, value) in header {
            let Some(offsets) = value.get("data_offsets").and_then(|o| o.as_array()) else { continue };
            let (start, end) = (offsets[0].as_u64().context("offset")?, offsets[1].as_u64().context("offset")?);
            if !name.contains(".experts.") && end > start {
                names.push((name, path.clone(), 8 + length as u64 + start, (end - start) as usize));
            }
        }
    }
    names.sort();
    anyhow::ensure!(names.len() == 2, "fixture has two dense tensors, found {}", names.len());
    // Both GPUs receive both tensors (a replica each, as head-split norms).
    let names: Vec<_> = names.iter().chain(&names).cloned().collect();
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    // Per GPU: one landed operand and one staged upload.
    let mut records = Records::default();
    let mut owned = Vec::new();
    for (gpu, pair) in names.chunks(2).enumerate() {
        let rects: Vec<_> = pair.iter().map(|(name, path, base, bytes)| Rect { name: name.clone(), path: path.clone(),
            base: *base, row_bytes: *bytes, rows: 0..1, cols: 0..*bytes }).collect();
        let allocations = devices[gpu].run(|| rects.iter().map(|r|
            crate::shared::memory::DeviceAllocation::new(&library, r.bytes().max(256))).collect::<Result<Vec<_>>>())?;
        records.operands.push(Operand { device: gpu as i32, buffer: allocations[0].buffer,
            lands: vec![(rects[0].clone(), 0)], bytes: rects[0].bytes() });
        records.steps.push(Step { device: gpu as i32, rects: vec![rects[1].clone()],
            jobs: vec![Job::Upload { at: 0, bytes: rects[1].bytes(), dst: allocations[1].buffer }] });
        owned.push((rects, allocations));
    }
    let replica = devices[0].run(|| crate::shared::memory::DeviceAllocation::new(&library, owned[1].0[0].bytes()))?;
    library.copy_h2d(replica.buffer, &vec![0xa5; replica.buffer.bytes])?;
    records.peer_copies.push(crate::shared::experts::rtx::dense::PeerCopy::new(
        owned[1].1[0].buffer, replica.buffer, replica.buffer.bytes)?);
    let mut untouched = vec![0; replica.buffer.bytes];
    devices[0].run(|| library.copy_d2h(&mut untouched, replica.buffer))?;
    assert_eq!(untouched, vec![0xa5; replica.buffer.bytes], "recording must not copy an unfilled source");
    let (plans, destinations) = plan_records(root.path(), records, &devices, 0)?;
    struct Observed<D> { inner: D, done: std::sync::mpsc::Sender<()> }
    impl<D: PushDestination<Delivered = ()>> PushDestination for Observed<D> {
        type Delivered = ();
        fn belongs_to(&self, device: Device<'_>) -> bool { self.inner.belongs_to(device) }
        fn logical_bytes(&self) -> usize { self.inner.logical_bytes() }
        fn device_bytes(&self) -> usize { self.inner.device_bytes() }
        fn check_fragment(&self, offset: u64, bytes: u64) -> Result<()> { self.inner.check_fragment(offset, bytes) }
        fn destination(&self, offset: usize, bytes: usize) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
            self.inner.destination(offset, bytes)
        }
        fn copy_validation(&mut self, offset: u64, bytes: &[u8]) -> Result<()> { self.inner.copy_validation(offset, bytes) }
        fn copied(&mut self, offset: u64, bytes: u64) -> Result<()> { self.inner.copied(offset, bytes) }
        fn done(self) -> Result<()> {
            self.inner.done()?;
            self.done.send(())?;
            Ok(())
        }
    }
    let (rank_done, completed) = std::sync::mpsc::channel();
    let dense = (plans, destinations.into_iter().map(|inner| Observed { inner, done: rank_done.clone() }).collect());
    drop(rank_done);
    let (allow_commit, commit_allowed) = std::sync::mpsc::channel();
    let sockets = tempfile::tempdir_in(fixture.directory())?;
    let socket = sockets.path().join("hub.sock").to_string_lossy().into_owned();
    let listener = Listener::bind(&socket)?;
    let push = CoordinatorPush::prepare_alone_at(root.path(), None, 2, None, Some(socket.clone()))?;
    let hub_root = root.path().to_path_buf();
    let hub_socket = socket.clone();
    let hub = std::thread::spawn(move || -> Result<()> {
        let listener = std::sync::Arc::new(listener);
        let accept = { let listener = listener.clone(); let hub_socket = hub_socket.clone();
            move || crate::shared::memory::push_testing::accept_within(&listener, &hub_socket, Duration::from_secs(120)) };
        let coordinator = accept()?;
        coordinator.set_timeout(Some(Duration::from_secs(60)))?;
        let (request, _) = coordinator.next_request()?.context("no Create")?;
        let Request::Create { ranks, files, .. } = request else { anyhow::bail!("expected Create") };
        anyhow::ensure!(ranks.iter().map(|r| r.rank).eq([0, 1]), "coordinator ranks 0 and 1 alone");
        coordinator.send_event(&Event::Created { group: 7, token: 11 })?;
        // Ranks join (on their own connections) while the plan uploads; the
        // driver starts once both have joined.
        let joins = std::thread::spawn(move || -> Result<Vec<_>> {
            let mut joined = Vec::new();
            for _ in 0..2 {
                let conn = accept()?;
                conn.set_timeout(Some(Duration::from_secs(60)))?;
                let (request, fd) = conn.next_request()?.context("rank did not join")?;
                let Request::Join { rank, slots, slot_bytes, .. } = request else { anyhow::bail!("expected Join") };
                conn.send_event(&Event::Joined)?;
                joined.push((rank, conn, File::from(fd.context("ring fd")?), slots, slot_bytes));
            }
            Ok(joined)
        });
        let (mut patterns, mut entries) = (Vec::new(), Vec::new());
        loop {
            match coordinator.next_request()?.context("plan upload ended")?.0 {
                Request::Patterns(p) => patterns.extend(p),
                Request::Entries(e) => entries.extend(e),
                Request::Start => break,
                other => anyhow::bail!("unexpected {other:?}"),
            }
        }
        coordinator.send_event(&Event::Planned(Default::default()))?;
        let plan = cuteafd_loader::push_load::plan::Plan { files, patterns, entries };
        let joined = joins.join().map_err(|_| anyhow::anyhow!("join acceptor panicked"))??;
        for (rank, conn, ring, slots, slot_bytes) in &joined {
            let mut runs = Vec::new();
            for entry in plan.entries.iter().filter(|e| e.rank == *rank) {
                let pattern = &plan.patterns[entry.pattern as usize];
                for (src, dst) in pattern.runs() { runs.push((entry.file, entry.src_base + src, entry.dst_base + dst, pattern.run)); }
            }
            runs.sort_unstable();
            let mut index = 0usize;
            let mut packer = slot::Packer::new(vec![0; *slot_bytes as usize]);
            let mut send = |bytes: Vec<u8>| -> Result<()> {
                let at = index % *slots as usize;
                ring.write_all_at(&bytes, at as u64 * u64::from(*slot_bytes))?;
                conn.send_event(&Event::Filled { slot: at as u32, len: bytes.len() as u32 })?;
                let (release, _) = conn.next_request()?.context("missing release")?;
                anyhow::ensure!(matches!(release, Request::Release { slot } if slot == at as u32), "release order");
                index += 1;
                Ok(())
            };
            for (file, src, dst, length) in runs {
                let mut run = vec![0u8; length as usize];
                File::open(hub_root.join(&plan.files[file as usize]))?.read_exact_at(&mut run, src)?;
                let (mut bytes, mut destination) = (&run[..], dst);
                while !bytes.is_empty() {
                    let n = packer.push(destination, bytes);
                    if n == 0 {
                        send(std::mem::replace(&mut packer, slot::Packer::new(vec![0; *slot_bytes as usize])).finish().0)?;
                        continue;
                    }
                    destination += n as u64;
                    bytes = &bytes[n..];
                }
            }
            if !packer.is_empty() { send(packer.finish().0)?; }
            conn.send_event(&Event::Done)?;
        }
        // Withhold group Done until both rank finishes and the replica-byte check.
        commit_allowed.recv_timeout(Duration::from_secs(60))?;
        coordinator.send_event(&Event::Done)?;
        // The driver keeps the group open until every rank initialized, then closes it.
        anyhow::ensure!(coordinator.next_request()?.is_none(), "the driver closes its group after the commit");
        Ok(())
    });
    let (pushed, observed) = push.receive_with(devices.to_vec(), None, 0, Some(Ok(dense)), || -> Result<()> {
        for _ in 0..2 { completed.recv_timeout(Duration::from_secs(60))?; }
        let checked = devices[0].run(|| -> Result<()> {
            let mut actual = vec![0; replica.buffer.bytes];
            library.copy_d2h(&mut actual, replica.buffer)?;
            anyhow::ensure!(actual == untouched, "rank Done must not copy before whole-group commit");
            Ok(())
        });
        // Release the protocol even if the assertion fails, so cleanup cannot deadlock.
        allow_commit.send(())?;
        checked
    });
    let served = hub.join().map_err(|_| anyhow::anyhow!("fake hub panicked"))?;
    served?;
    observed?;
    let pushed = pushed?;
    assert_eq!(pushed.map(|ranks| ranks.len()), Some(2), "both coordinator ranks commit");
    let mut expected_replica = vec![0; replica.buffer.bytes];
    owned[1].0[0].read(&mut expected_replica)?;
    let mut actual_replica = vec![0; replica.buffer.bytes];
    devices[0].run(|| library.copy_d2h(&mut actual_replica, replica.buffer))?;
    assert_eq!(actual_replica, expected_replica, "group-post-commit peer copy reads the other rank's landed bytes");
    for (gpu, (rects, allocations)) in owned.iter().enumerate() {
        for (rect, allocation) in rects.iter().zip(allocations) {
            let mut expected = vec![0; rect.bytes()];
            rect.read(&mut expected)?;
            let mut actual = vec![0; rect.bytes()];
            devices[gpu].run(|| library.copy_d2h(&mut actual,
                cuteafd_ffi::CuteafdDeviceBuffer { bytes: rect.bytes(), ..allocation.buffer }))?;
            assert_eq!(actual, expected, "GPU {gpu} {}", rect.name);
        }
    }
    Ok(())
}

/// Coordinator EXL3 expert layers push byte for byte as the mount reads them,
/// as TP2 halves over two GPUs (world 2) and as whole layers on one (world 1,
/// the coordinator-local residency).
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn coordinator_exl3_layers_push_byte_exact_at_world_one_and_two() -> Result<()> {
    use crate::shared::experts::layer::ExpertLayer;
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    for device in 0..2 { fixture.configure_memory(&library, device, 64 << 30)?; }
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (tensors, _) = qwen4_exl3(2, 4);
    let tensors: Vec<_> = tensors.into_iter().filter(|t| !t.0.contains(".experts.") || t.0.contains(".experts.0.")).collect();
    let mut config = qwen4_config(2);
    config["text_config"]["num_experts"] = serde_json::json!(1);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    write_snapshot(root.path(), &config, &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    for tensor in catalog.tensors() {
        let image: Vec<u8> = if tensor.metadata.name.ends_with(".mcg") { 0xcbac1fedu32.to_le_bytes().to_vec() }
            else if tensor.metadata.name.ends_with(".suh") || tensor.metadata.name.ends_with(".svh") {
                [0u8, 0x3c].into_iter().cycle().take(tensor.metadata.byte_length as usize).collect()
            } else { (0..tensor.metadata.byte_length).map(|i| (i as u8).wrapping_mul(13) ^ 0x5a).collect() };
        File::options().write(true).open(root.path().join(&tensor.shard))?.write_all_at(&image, tensor.metadata.byte_offset)?;
    }
    let records = crate::shared::experts::rtx::dense::Records::default();
    let sockets = tempfile::tempdir_in(fixture.directory())?;
    let whole = [ExpertLayer::BackboneFull { layer: 0 }, ExpertLayer::BackboneFull { layer: 1 }];
    let plans: Vec<(CoordinatorExpertPlan, usize)> = vec![
        (CoordinatorExpertPlan::whole(whole.iter().map(|&s| crate::shared::experts::exl3::planned_residency(&catalog, s))
            .collect::<Result<Vec<_>>>()?, usize::MAX)?, 0),
        (CoordinatorExpertPlan::new(&catalog, &[0, 1], 0, usize::MAX)?, 0),
        (CoordinatorExpertPlan::new(&catalog, &[0, 1], 1, usize::MAX)?, 1),
    ];
    for (plan, gpu) in plans {
        let world = plan.residencies[0].world;
        let pushed = receive_pushed(Device { library: &library, id: gpu as i32 }, Some(&catalog), root.path(), sockets.path(),
            &records, Some(plan), 0, gpu)?.exl3()?;
        assert_eq!(pushed.len(), 2, "world {world}");
        for weight in &pushed {
            assert_eq!(weight.layout.world, world);
            let mut scratch = vec![0; weight.layout.scratch_bytes(&catalog)?];
            for (index, load) in weight.layout.loads.iter().enumerate() {
                let mut expected = vec![0; load.bytes];
                weight.layout.read_into(&catalog, index, &mut expected, &mut scratch)?;
                for &(buffer, offset) in &load.destinations {
                    let spec = &weight.layout.buffers[buffer];
                    let buffer = weight.buffer(&spec.name)?;
                    let mut actual = vec![0; load.bytes];
                    // SAFETY: the load's destination range lies inside this published buffer.
                    let view = cuteafd_ffi::CuteafdDeviceBuffer { bytes: load.bytes,
                        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() }, ..buffer };
                    Device { library: &library, id: gpu as i32 }.run(|| library.copy_d2h(&mut actual, view))?;
                    assert_eq!(actual, expected, "world {world} {}", load.tensor);
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn combined_exlr8_tp2_rank_is_mount_exact_and_drains_on_rejection() -> Result<()> {
    use crate::shared::{experts::exlr8::{Exlr8Weights, view}, memory::push_exlr8::tests::fixture_catalog};
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    for device in 0..2 { fixture.configure_memory(&library, device, 64 << 30)?; }
    std::fs::write(fixture.directory().join("events"), b"")?;
    for padded in [false, true] {
        let root = tempfile::tempdir_in(fixture.directory())?;
        cuteafd_loader::plan::testing::exlr8::fixture(root.path(), padded);
        let catalog = fixture_catalog(root.path())?;
        for gpu in 0..2 {
            for fail in [false, true] {
                let device = Device { library: &library, id: gpu as i32 };
                let (rank, world, base) = (6 + gpu, 8, 256u64);
                let plan = CoordinatorExpertPlan::exlr8(&catalog, "mixed", &[3], 2, gpu, usize::MAX)?;
                let checked_tp2 = plan.checked(&catalog, root.path(), world, rank)?;
                let mut arenas = vec![0; world];
                arenas[rank] = base + plan.logical_bytes();
                let mut builder = Builder::new(root.path(), arenas);
                let dense_source = root.path().join("dense/exlr8-dense-000.safetensors");
                let dense_bytes = std::fs::read(&dense_source)?[..32].to_vec();
                builder.contiguous(&cuteafd_loader::push_load::builder::Source {
                    path: dense_source, offset: 0, bytes: dense_bytes.len() as u64 }, 0, dense_bytes.len() as u64,
                    cuteafd_loader::push_load::builder::Destination { rank: rank as u32, offset: 0,
                        bytes: dense_bytes.len() as u64 })?;
                plan.append_at(&mut builder, &catalog, rank as u32, base)?;
                let combined = builder.finish()?;
                let packed = pack(&combined, root.path(), rank as u32, DEFAULT_SLOT_BYTES)?;
                let socket = root.path().join("c.sock").to_string_lossy().into_owned();
                let listener = Listener::bind(&socket)?;
                let (mut coordinator, mut worker) = control_pair()?;
                let events_path = fixture.directory().join("events");
                let before = fixture.events()?.len();
                let server_socket = socket.clone();
                let server = std::thread::spawn(move || -> Result<()> {
                    let conn = crate::shared::memory::push_testing::accept_within(&listener, &server_socket,
                        Duration::from_secs(120))?;
                    conn.set_timeout(Some(Duration::from_secs(10)))?;
                    let (request, fd) = conn.next_request()?.context("EXLR8 rank did not join")?;
                    assert!(matches!(request, Request::Join { rank: r, .. } if r == rank as u32));
                    let file = File::from(fd.context("EXLR8 ring fd absent")?);
                    conn.send_event(&Event::Joined)?;
                    assert_eq!(read_control(&coordinator)?["message"], "joined");
                    for (index, bytes) in packed.into_iter().enumerate() {
                        let slot = index % 8;
                        file.write_all_at(&bytes, slot as u64 * DEFAULT_SLOT_BYTES as u64)?;
                        conn.send_event(&Event::Filled { slot: slot as u32, len: bytes.len() as u32 })?;
                        assert!(matches!(conn.next_request()?.context("missing release")?.0,
                            Request::Release { slot: s } if s == slot as u32));
                    }
                    conn.send_event(&Event::Done)?;
                    assert_eq!(read_control(&coordinator)?["success"], true);
                    write_control(&mut coordinator, &serde_json::json!({"message":"commit", "success":!fail}))?;
                    if fail {
                        assert_eq!(read_control(&coordinator)?["message"], "drained");
                        assert_eq!(std::fs::read_to_string(events_path)?[before..].matches('d').count(), 1,
                            "EXLR8 arena retained at drained barrier");
                        write_control(&mut coordinator, &serde_json::json!({"message":"mount"}))?;
                    }
                    Ok(())
                });
                let delivered = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
                let dense = HostDense { image: vec![0; 32], planned: vec![(0, 32)], delivered: delivered.clone(),
                    device: gpu as i32 };
                let client = PushClient::at(socket, Duration::from_secs(10))?;
                let weights = receive_rank(device, Some((&catalog, plan, &checked_tp2)),
                    Some(DenseRank { destination: dense, base }), rank, &client, &mut worker, (7, 11), 8, DEFAULT_SLOT_BYTES)?;
                server.join().map_err(|_| anyhow::anyhow!("EXLR8 fake source panicked"))??;
                if fail { assert!(weights.is_none()); continue; }
                assert_eq!(*delivered.lock().unwrap(), dense_bytes);
                let weights = weights.context("EXLR8 push fell back")?.exlr8()?;
                assert_eq!(weights.len(), 1);
                device.run(|| {
                    let pushed = &weights[0];
                    let mount = Exlr8Weights::load(&library, pushed.layout.clone(), usize::MAX)?;
                    let mut actual = vec![0; pushed.image().bytes];
                    let mut expected = vec![0; mount.image().bytes];
                    library.copy_d2h(&mut actual, pushed.image())?;
                    library.copy_d2h(&mut expected, mount.image())?;
                    // Only absolute descriptor addresses relocate with the arena.
                    let (pointers, _) = pushed.layout.descriptors(pushed.image().ptr as u64)?;
                    let pointer_image = pointers.into_iter().flat_map(u64::to_le_bytes).collect::<Vec<_>>();
                    let at = pushed.layout.pointers_offset;
                    expected[at..at + pointer_image.len()].copy_from_slice(&pointer_image);
                    // All bytes, including signs, compact maps and padding, must agree.
                    assert_eq!(actual, expected);
                    for (offset, bytes) in [(pushed.layout.bits_offset, pushed.bits()?.bytes),
                        (pushed.layout.signs_offsets[0], pushed.layout.hidden)] {
                        let mut image = vec![0; bytes];
                        library.copy_d2h(&mut image, view(pushed.image(), offset, bytes)?)?;
                        assert_eq!(image, expected[offset..offset + bytes]);
                    }
                    Ok(())
                })?;
                device.run(|| { drop(weights); Ok(()) })?;
            }
        }
    }
    Ok(())
}
